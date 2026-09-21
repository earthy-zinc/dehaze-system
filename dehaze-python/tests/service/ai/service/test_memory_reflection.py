"""反思整合记忆回归：user_id 透传 + ES 向量索引同步

identity-forward 将 _score_importance 的 user_id 改为必填后，reflect_and_consolidate
存在漏传（TypeError 使反思定时任务对所有用户失败）；反思/合并产生的记忆此前不写
ES，导致永远无法被检索注入层命中（§7.7 MySQL 写入后同步 ES）。
"""

from types import SimpleNamespace

from app.service.ai.service import memory_extraction
from app.service.ai.service.memory_extraction import reflect_and_consolidate


def _install(monkeypatch, insight_content="用户每周一需要周报"):
    captured = {}

    async def _llm_text(db, model_id, prompt, system_prompt, max_tokens, user_id=None):
        return f'```json\n[{{"type": "procedural", "content": "{insight_content}"}}]\n```'

    async def _score(db, model_id, content, user_id=None):
        captured.setdefault("user_ids", []).append(user_id)
        return 60

    async def _list_recent(db, user_id, since):
        return [SimpleNamespace(content="用户今天说喜欢简洁风格")]

    created = []

    async def _exists(db, user_id, memory_type, content):
        return False

    async def _create(db, memory):
        created.append(memory)
        return SimpleNamespace(id=777)

    async def _sync_memory(doc):
        captured["synced"] = doc

    monkeypatch.setattr(memory_extraction, "_llm_text", _llm_text)
    monkeypatch.setattr(memory_extraction, "_score_importance", _score)
    monkeypatch.setattr(
        memory_extraction.ai_memory_repository,
        "list_recent_episodic",
        _list_recent,
    )
    monkeypatch.setattr(memory_extraction.ai_memory_repository, "exists_active_content", _exists)
    monkeypatch.setattr(memory_extraction.ai_memory_repository, "create", _create)
    monkeypatch.setattr(memory_extraction, "sync_memory", _sync_memory)
    return captured


async def test_reflection_passes_user_id_to_importance(monkeypatch):
    """user_id 必填漏传会让反思任务对所有用户 TypeError，此处锁定透传"""
    captured = _install(monkeypatch)
    saved = await reflect_and_consolidate(object(), 42, "test-model")
    assert saved == 1
    assert captured["user_ids"] == [42]


async def test_reflection_syncs_memory_to_es(monkeypatch):
    """反思洞察写入后同步 ES，否则永远无法被检索注入命中"""
    captured = _install(monkeypatch)
    await reflect_and_consolidate(object(), 42, "test-model")
    doc = captured["synced"]
    assert doc["id"] == 777
    assert doc["user_id"] == 42
    assert doc["memory_type"] == "procedural"
