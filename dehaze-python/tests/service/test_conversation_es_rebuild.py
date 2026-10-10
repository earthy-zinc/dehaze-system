"""会话 ES 读模型重建/对账能力单测

覆盖 conversation_search_service.rebuild_conversation_index：
重同步修正陈旧 deleted 标记、清理孤儿文档、重复执行幂等。
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace

from app.infrastructure.es.es_client import es_client
from app.service.ai.service import conversation_search_service as svc


class _FakeEs:
    """内存 ES 桩：仅实现重建/对账用到的文档读写原语"""

    def __init__(self, docs: dict[str, dict]) -> None:
        self.docs = docs

    async def index_doc(self, index_name: str, doc_id: str, doc: dict) -> bool:
        self.docs[doc_id] = doc
        return True

    async def list_doc_ids(self, index_name: str) -> list[str]:
        return list(self.docs)

    async def delete_doc(self, index_name: str, doc_id: str) -> bool:
        self.docs.pop(doc_id, None)
        return True


def _patch(monkeypatch, fake: _FakeEs, conversations: dict[int, dict]) -> None:
    """注入内存 ES 桩与会话表桩，使 sync_conversation_to_es 走真实聚合写入路径"""

    @asynccontextmanager
    async def _session():
        yield object()

    monkeypatch.setattr(es_client, "index_doc", fake.index_doc)
    monkeypatch.setattr(es_client, "list_doc_ids", fake.list_doc_ids)
    monkeypatch.setattr(es_client, "delete_doc", fake.delete_doc)
    monkeypatch.setattr(svc, "get_db_session", _session)

    async def list_all_ids(db):
        return list(conversations)

    async def get_by_id(db, conv_id, *, with_deleted=False):
        row = conversations[conv_id]
        return SimpleNamespace(
            user_id=row["user_id"],
            title=row["title"],
            status=row["status"],
            deleted=row["deleted"],
        )

    async def list_by_conversation(db, conv_id, page, size, **kwargs):
        return [], 0

    monkeypatch.setattr(svc.ai_conversation_repository, "list_all_ids", list_all_ids)
    monkeypatch.setattr(svc.ai_conversation_repository, "get_by_id", get_by_id)
    monkeypatch.setattr(svc.ai_message_repository, "list_by_conversation", list_by_conversation)


async def test_resync_fixes_stale_deleted_marker(monkeypatch):
    """MySQL 已软删的会话在 ES 仍是 deleted=0（存量漂移）→ 重同步按 MySQL 修正"""
    conversations = {
        1: {"user_id": 1, "title": "活跃", "status": 1, "deleted": 0},
        2: {"user_id": 2, "title": "已删", "status": 1, "deleted": 2},
    }
    fake = _FakeEs(
        {
            "1": {"id": 1, "user_id": 1, "status": 1, "deleted": 0},
            "2": {"id": 2, "user_id": 2, "status": 1, "deleted": 0},
        }
    )
    _patch(monkeypatch, fake, conversations)

    summary = await svc.rebuild_conversation_index()

    assert summary == {"resynced": 2, "deleted_orphans": 0}
    assert fake.docs["2"]["deleted"] == 2
    assert fake.docs["1"]["deleted"] == 0


async def test_prunes_orphan_docs(monkeypatch):
    """ES 中在 MySQL 已不存在的文档（孤儿）应被删除"""
    conversations = {1: {"user_id": 1, "title": "活跃", "status": 1, "deleted": 0}}
    fake = _FakeEs({"1": {"id": 1, "user_id": 1, "status": 1, "deleted": 0}, "999": {"id": 999}})
    _patch(monkeypatch, fake, conversations)

    summary = await svc.rebuild_conversation_index()

    assert summary == {"resynced": 1, "deleted_orphans": 1}
    assert set(fake.docs) == {"1"}


async def test_rebuild_is_idempotent(monkeypatch):
    """重复执行幂等：文档集合稳定，漂移修正结果保持"""
    conversations = {
        1: {"user_id": 1, "title": "活跃", "status": 1, "deleted": 0},
        2: {"user_id": 2, "title": "已删", "status": 2, "deleted": 2},
    }
    fake = _FakeEs({"1": {"id": 1, "deleted": 0}, "2": {"id": 2, "deleted": 0}, "999": {"id": 999}})
    _patch(monkeypatch, fake, conversations)

    # 首轮清理漂移与孤儿；此后每轮均为无副作用的重写（幂等）
    first = await svc.rebuild_conversation_index()
    second = await svc.rebuild_conversation_index()
    third = await svc.rebuild_conversation_index()

    assert first == {"resynced": 2, "deleted_orphans": 1}
    assert second == third == {"resynced": 2, "deleted_orphans": 0}
    assert set(fake.docs) == {"1", "2"}
    assert fake.docs["2"]["deleted"] == 2
    assert fake.docs["2"]["status"] == 2
