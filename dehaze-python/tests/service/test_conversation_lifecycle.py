import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.schema.ai_conversation import ConversationUpdate
from app.repository.ai_conversation_repository import AiConversationRepository
from app.repository.ai_message_repository import AiMessageRepository
from app.service import ai_conversation_service as m
from app.service.ai_conversation_service import AiConversationService

pytestmark = pytest.mark.requires_db


def _stream_text(chunk: str | bytes | memoryview) -> str:
    """StreamingResponse.body_iterator 产出的 Content 为 str|bytes|memoryview，归一为文本。"""
    return chunk if isinstance(chunk, str) else bytes(chunk).decode()


def _conv(**kw):
    base = {
        "id": 1,
        "user_id": 1,
        "title": "测试会话",
        "model": "m1",
        "agent_code": "default",
        "agent_version": 1,
        "summary": None,
        "system_prompt": None,
        "model_config": None,
        "api_key_id": None,
        "message_count": 3,
        "last_message_at": None,
        "current_branch_message_id": 10,
        "last_read_message_id": None,
        "pinned": 0,
        "pinned_at": None,
        "delete_time": None,
        "title_source": "auto",
        "status": 1,
        # 列为 NOT NULL default=1，脱离 session 实例化不会应用列默认值，显式补上
        "suggestions_enabled": 1,
        "create_time": datetime(2026, 1, 1),
        "update_time": datetime(2026, 1, 1),
        "deleted": 0,
    }
    base.update(kw)
    # 真实 ORM 实体（可脱离 session 实例化）：字段与生产 SysAiConversation 一致
    return SysAiConversation(**base)


def _msg(**kw):
    base = {
        "id": 1,
        "conversation_id": 1,
        "parent_message_id": None,
        "role": "user",
        "content": "你好",
        "status": 2,
        "deleted": 0,
        "create_time": datetime(2026, 1, 1),
    }
    base.update(kw)
    return SimpleNamespace(**base)


async def _async(*args, **kwargs):
    return None


async def _export_chain(db, cid, tail, limit=None):
    return [
        _msg(id=1, role="user", content="帮我分析一下雾霾的形成原因"),
        _msg(id=2, role="assistant", content="好的，我从排放源、气象条件两方面梳理"),
        _msg(id=3, role="tool", content="工具返回：查询到监测数据"),
        _msg(id=4, role="assistant", content=""),
    ]


def _set_pin(state):
    async def _stub(_db, cid, pinned, at):
        state["pinned"] = pinned
        state["at"] = at

    return _stub


async def _es_empty(user_id, query, *, status, page, size):
    return [], 0


def _paginate_capturing(captured):
    async def _stub(_db, uid, p, s, status=None):
        captured["status"] = status
        return [_conv(id=1)], 1

    return _stub


class _ConvRepo:
    async def get_by_id_and_user(self, db, cid, uid):
        return _conv(id=cid)

    async def paginate_user_conversations(self, db, uid, p, s, status=None):
        return [], 0

    async def paginate_all_with_keyword(self, db, p, s, kw, status=None):
        return [], 0

    async def paginate_all_conversations(self, db, p, s, status=None):
        return [], 0

    async def update_status(self, db, ids, st):
        return None

    async def soft_delete_by_ids(self, db, ids):
        return 1

    async def count_active_pinned(self, db, uid):
        return 0

    async def set_pinned(self, db, cid, pinned, at):
        return None

    async def get_in_trash(self, db, cid, uid, win):
        return _conv(id=cid)

    async def restore_by_ids(self, db, ids):
        return 1

    async def paginate_trash(self, db, uid, p, s, win):
        return [], 0

    async def get_by_ids_and_user(self, db, ids, uid):
        return [_conv(id=i) for i in ids]

    async def mark_read(self, db, cid, mid):
        return None


class _MsgRepo:
    async def get_last_message_id(self, db, cid):
        return 10

    async def count_messages_after(self, db, cid, aid):
        return 0

    async def find_latest_ids_by_keyword(self, db, conv_ids, keyword):
        return {}

    async def get_chain_by_id(self, db, cid, tail, limit=None):
        return []


def _make_service(*, conv=None, msg=None):
    """构造 AiConversationService：conv/msg 为 {方法名: 桩}，注入到对应仓储桩"""
    conv_repo = _ConvRepo()
    msg_repo = _MsgRepo()
    for name, fn in (conv or {}).items():
        setattr(conv_repo, name, fn)
    for name, fn in (msg or {}).items():
        setattr(msg_repo, name, fn)
    return AiConversationService(
        # 测试替身：_ConvRepo/_MsgRepo 为多方法结构型桩（且 _make_service 对其动态 setattr 覆写），
        # 子类化会触发方法覆写告警
        ai_conversation_repository=cast(AiConversationRepository, conv_repo),  # 替身：结构型桩
        ai_message_repository=cast(AiMessageRepository, msg_repo),  # 替身：结构型桩 + 动态覆写
    )


class TestBatchOperate:
    async def test_archive_sets_status_2(self, db):
        updated = []

        async def upd(_db, ids, st):
            updated.append((ids, st))

        svc = _make_service(conv={"update_status": upd})
        count = await svc.batch_operate(db, 1, "archive", [1, 2])
        assert count == 2
        assert updated == [([1], 2), ([2], 2)]

    async def test_batch_delete_requires_confirm(self, db):
        svc = _make_service()
        with pytest.raises(BusinessException):
            await svc.batch_operate(db, 1, "delete", [1], confirm=False)

    async def test_batch_rollback_on_failure(self, db):
        calls = []

        async def upd(_db, ids, st):
            calls.append((ids, st))

        async def get_owned(_db, cid, uid):
            if cid == 2:
                raise BusinessException(m.ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
            return _conv(id=cid, status=1)

        svc = _make_service(conv={"get_by_id_and_user": get_owned, "update_status": upd})
        with pytest.raises(BusinessException) as exc:
            await svc.batch_operate(db, 1, "archive", [1, 2])
        assert "会话不存在" in str(exc.value)
        assert calls == [([1], 2)]

    async def test_batch_restore_invalid_state_rolls_back(self, db):
        async def get_owned(_db, cid, uid):
            if cid == 2:
                return _conv(id=cid, status=1)
            return _conv(id=cid, status=2)

        svc = _make_service(conv={"get_by_id_and_user": get_owned})
        with pytest.raises(BusinessException):
            await svc.batch_operate(db, 1, "restore", [1, 2])


class TestPinLimit:
    async def test_pin_exceeds_limit_raises(self, db):
        async def count_pinned(_db, uid):
            return m.PINNED_CONVERSATION_LIMIT

        svc = _make_service(conv={"count_active_pinned": count_pinned})
        with pytest.raises(BusinessException) as exc:
            await svc.pin_conversation(db, 1, 1)
        assert exc.value.code.code == "A0501"

    async def test_pin_sets_pinned_at(self, db):
        """置顶写入 pinned_at：pin_conversation 内部 commit 后需 refresh 会话行。"""
        conv = SysAiConversation(user_id=1, title="会话", model="m1", pinned=0, status=1)
        db.add(conv)
        await db.flush()
        state = {}

        async def get_owned(_db, cid, uid):
            return conv

        svc = _make_service(conv={"get_by_id_and_user": get_owned, "set_pinned": _set_pin(state)})
        result = await svc.pin_conversation(db, conv.id, 1)
        assert state["pinned"] == 1
        assert state["at"] is not None
        assert result.pinned == 1

    async def test_unpin_clears_pinned_at(self, db):
        state = {}
        svc = _make_service(conv={"set_pinned": _set_pin(state)})
        result = await svc.unpin_conversation(db, 1, 1)
        assert state["pinned"] == 0
        assert state["at"] is None
        assert result.pinned == 0

    async def test_patch_pinned_keeps_existing_pinned_at(self, db):
        """已置顶会话再次 PATCH pinned=1：不刷新 pinned_at，避免打乱置顶排序"""
        original = datetime(2026, 1, 1, 10, 0, 0)
        conv = SysAiConversation(
            user_id=1, title="已置顶", model="m1", pinned=1, pinned_at=original, status=1
        )
        db.add(conv)
        await db.flush()

        result = await AiConversationService().update_conversation(
            db, conv.id, 1, ConversationUpdate(pinned=1)
        )
        assert result.pinned_at == original
        await db.refresh(conv)
        assert conv.pinned_at == original

    async def test_patch_pin_sets_pinned_at(self, db):
        conv = SysAiConversation(user_id=1, title="会话", model="m1", pinned=0, status=1)
        db.add(conv)
        await db.flush()

        result = await AiConversationService().update_conversation(
            db, conv.id, 1, ConversationUpdate(pinned=1)
        )
        assert result.pinned == 1
        assert result.pinned_at is not None

    async def test_concurrent_pin_respects_limit(self, db):
        """并发置顶同一用户的不同会话：上限校验与写入在用户级锁内完成，不会突破上限

        无锁时两个请求都会读到 9 条置顶并双双通过校验（check-then-act），最终置顶 11 条。
        """
        state = {"pinned": m.PINNED_CONVERSATION_LIMIT - 1}

        async def count_pinned(_db, uid):
            await asyncio.sleep(0)
            return state["pinned"]

        async def set_pinned(_db, cid, pinned, at):
            state["pinned"] += 1

        svc = _make_service(conv={"count_active_pinned": count_pinned, "set_pinned": set_pinned})
        results = await asyncio.gather(
            svc.pin_conversation(AsyncMock(), 1, 1),
            svc.pin_conversation(AsyncMock(), 2, 1),
            return_exceptions=True,
        )
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, BusinessException)]
        assert len(successes) == 1
        assert len(failures) == 1
        assert state["pinned"] == m.PINNED_CONVERSATION_LIMIT


class TestRestoreWindow:
    async def test_restore_within_window(self, db):
        restored = []

        async def restore(_db, ids):
            restored.append(ids)

        async def in_trash(_db, cid, uid, win):
            return _conv(id=cid)

        svc = _make_service(conv={"restore_by_ids": restore, "get_in_trash": in_trash})
        result = await svc.restore_conversation(db, 1, 1)
        assert restored == [[1]]
        assert result.id == 1

    async def test_restore_outside_window_raises(self, db):
        async def in_trash(_db, cid, uid, win):
            return None

        svc = _make_service(conv={"get_in_trash": in_trash})
        with pytest.raises(BusinessException) as exc:
            await svc.restore_conversation(db, 1, 1)
        assert "恢复窗口" in str(exc.value)

    async def test_trash_passes_window(self, db):
        captured = {}

        async def paginate(_db, uid, p, s, win):
            captured["win"] = win
            return [_conv(id=1)], 1

        svc = _make_service(conv={"paginate_trash": paginate})
        await svc.list_trash(db, 1, 1, 10)
        assert (datetime.now() - captured["win"]).days >= 29


class TestReadAndUnread:
    async def test_mark_read_sets_last_message_id(self, db):
        state = {}

        async def mark(_db, cid, mid):
            state["cid"] = cid
            state["mid"] = mid

        async def last_msg(_db, cid):
            return 10

        svc = _make_service(
            conv={"mark_read": mark},
            msg={"get_last_message_id": last_msg},
        )
        await svc.mark_read(db, 1, 1)
        assert state == {"cid": 1, "mid": 10}

    async def test_unread_count_computed(self, db):
        async def count_after(_db, cid, aid):
            return 5

        svc = _make_service(msg={"count_messages_after": count_after})
        result = await svc._to_result(db, _conv(id=1, last_read_message_id=3))
        assert result.unread_count == 5

    async def test_unread_count_defaults_to_message_count(self, db):
        svc = _make_service()
        result = await svc._to_result(db, _conv(id=1, last_read_message_id=None, message_count=7))
        assert result.unread_count == 7


class TestExport:
    async def test_export_markdown_filters_non_dialogue(self, db):
        svc = _make_service(msg={"get_chain_by_id": _export_chain})
        resp = await svc.export_conversation(db, 1, 1, "markdown")
        chunks = [chunk async for chunk in resp.body_iterator]
        body = "".join(_stream_text(chunk) for chunk in chunks)
        assert "# 测试会话" in body
        assert "## 用户" in body
        assert "## 助手" in body
        assert "帮我分析一下雾霾的形成原因" in body
        assert "工具返回：查询到监测数据" not in body
        assert resp.headers["Content-Disposition"].startswith(
            'attachment; filename="conversation_1.md"'
        )

    async def test_export_json_keeps_dialogue_messages_only(self, db):
        svc = _make_service(msg={"get_chain_by_id": _export_chain})
        resp = await svc.export_conversation(db, 1, 1, "json")
        chunks = [chunk async for chunk in resp.body_iterator]
        body = "".join(_stream_text(chunk) for chunk in chunks)
        data = json.loads(body)
        assert data["conversation"]["title"] == "测试会话"
        roles = [msg["role"] for msg in data["messages"]]
        assert roles == ["user", "assistant", "assistant"]
        assert all(r in ("user", "assistant") for r in roles)
        assert resp.headers["Content-Disposition"].startswith(
            'attachment; filename="conversation_1.json"'
        )


class TestESSyncDefer:
    """ES 读模型同步延迟到事务提交后：Service 层只登记提交后回调，不直接触 ES"""

    async def test_restore_registers_post_commit_sync(self, db, monkeypatch):
        from app.database import run_after_commit_callbacks
        from app.service.ai.service import conversation_search_service

        called = []

        async def fake_sync(conv_id):
            called.append(conv_id)

        monkeypatch.setattr(conversation_search_service, "sync_conversation_to_es", fake_sync)
        svc = _make_service()
        await svc.restore_conversation(db, 1, 1)
        assert called == []
        await run_after_commit_callbacks(db)
        assert called == [1]

    async def test_batch_archive_registers_sync_for_each(self, db, monkeypatch):
        from app.database import run_after_commit_callbacks
        from app.service.ai.service import conversation_search_service

        called = []

        async def fake_sync(conv_id):
            called.append(conv_id)

        monkeypatch.setattr(conversation_search_service, "sync_conversation_to_es", fake_sync)
        svc = _make_service()
        await svc.batch_operate(db, 1, "archive", [1, 2])
        await run_after_commit_callbacks(db)
        assert sorted(called) == [1, 2]


class TestESList:
    async def test_es_search_passes_status_and_pagination(self, db, monkeypatch):
        captured = {}

        async def search(user_id, query, *, status, page, size):
            captured.update(status=status, page=page, size=size)
            return [1, 2], 2

        monkeypatch.setattr(m, "search_conversations", search)
        svc = _make_service(conv={"get_by_ids_and_user": _default_get_by_ids_and_user})
        result = await svc.list_conversations(db, 1, 2, 5, keyword="雾", status=1)
        assert captured == {"status": 1, "page": 2, "size": 5}
        assert result.total == 2
        assert len(result.list) == 2

    async def test_es_search_defaults_active_status(self, db, monkeypatch):
        captured = {}

        async def search(user_id, query, *, status, page, size):
            captured["status"] = status
            return [], 0

        monkeypatch.setattr(m, "search_conversations", search)
        svc = _make_service()
        await svc.list_conversations(db, 1, 1, 10, keyword="x")
        assert captured["status"] == 1

    async def test_es_sort_pinned_first_then_time(self):
        now = datetime.now()
        convs = [
            _conv(id=1, pinned=0, pinned_at=None, last_message_at=now),
            _conv(id=2, pinned=1, pinned_at=now - timedelta(hours=2), last_message_at=None),
            _conv(id=3, pinned=1, pinned_at=now, last_message_at=None),
        ]
        svc = _make_service()
        result = svc._sort_conversations(convs)
        assert [c.id for c in result] == [3, 2, 1]

    async def test_keyword_es_empty_returns_empty_page(self, db, monkeypatch):
        """ES 必选：keyword 搜索 ES 无命中时直接返回空页，不降级 MySQL"""
        called = {}
        monkeypatch.setattr(m, "search_conversations", _es_empty)
        svc = _make_service(conv={"paginate_user_conversations": _paginate_capturing(called)})
        result = await svc.list_conversations(db, 1, 1, 10, keyword="雾", status=2)
        assert result.list == []
        assert result.total == 0
        assert "status" not in called

    async def test_status_all_passes_none_to_es(self, db, monkeypatch):
        es_status = {}

        async def search(user_id, query, *, status, page, size):
            es_status["status"] = status
            return [1], 1

        monkeypatch.setattr(m, "search_conversations", search)
        svc = _make_service(conv={"get_by_ids_and_user": _default_get_by_ids_and_user})
        await svc.list_conversations(db, 1, 1, 10, keyword="雾", status=0)
        assert es_status["status"] is None

    async def test_status_all_without_keyword_passes_none_to_mysql(self, db):
        captured = {}
        svc = _make_service(conv={"paginate_user_conversations": _paginate_capturing(captured)})
        await svc.list_conversations(db, 1, 1, 10, status=0)
        assert captured["status"] is None


async def _default_get_by_ids_and_user(db, ids, uid):
    return [_conv(id=i) for i in ids]
