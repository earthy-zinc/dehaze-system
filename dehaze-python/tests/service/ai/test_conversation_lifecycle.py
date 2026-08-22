import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from app.core.exceptions import BusinessException
from app.service import ai_conversation_service as m
from app.service.ai_conversation_service import ai_conversation_service
from tests.stubs import StubAsyncSession


def _conv(**kw):
    base = dict(
        id=1,
        user_id=1,
        title="测试会话",
        model="m1",
        agent_code="default",
        agent_version=1,
        summary=None,
        system_prompt=None,
        model_config=None,
        api_key_id=None,
        message_count=3,
        last_message_at=None,
        current_branch_message_id=10,
        last_read_message_id=None,
        pinned=0,
        pinned_at=None,
        delete_time=None,
        title_source="auto",
        status=1,
        create_time=datetime(2026, 1, 1),
        update_time=datetime(2026, 1, 1),
        deleted=0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _msg(**kw):
    base = dict(
        id=1,
        conversation_id=1,
        parent_message_id=None,
        role="user",
        content="你好",
        status=2,
        deleted=0,
        create_time=datetime(2026, 1, 1),
    )
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


def _patch(
    monkeypatch,
    *,
    get_owned=None,
    update_status=None,
    count_pinned=None,
    set_pinned=None,
    get_in_trash=None,
    restore=None,
    paginate_trash=None,
    get_last_msg=None,
    mark_read=None,
    count_after=None,
    search=None,
    paginate_user_conversations=None,
    get_chain=None,
):
    async def _default_get_owned(db, cid, uid):
        return _conv(id=cid)

    async def _default_paginate(db, uid, p, s, status=None):
        return [], 0

    async def _default_trash(db, uid, p, s, win):
        return [], 0

    async def _default_get_by_ids(db, uid, ids):
        return [_conv(id=i) for i in ids]

    async def _default_get_in_trash(db, cid, uid, win):
        return _conv(id=cid)

    async def _default_search(*args, **kwargs):
        return [1], 1

    async def _default_soft_delete(db, ids):
        return 1

    async def _default_count_pinned(db, uid):
        return 0

    async def _default_restore(db, ids):
        return 1

    async def _default_last_msg(db, cid):
        return 10

    async def _default_count_after(db, cid, aid):
        return 0

    monkeypatch.setattr(
        m.ai_conversation_repository, "get_by_id_and_user", get_owned or _default_get_owned
    )
    monkeypatch.setattr(
        m.ai_conversation_repository,
        "paginate_user_conversations",
        paginate_user_conversations or _default_paginate,
    )
    monkeypatch.setattr(m.ai_conversation_repository, "update_status", update_status or _async)
    monkeypatch.setattr(m.ai_conversation_repository, "soft_delete_by_ids", _default_soft_delete)
    monkeypatch.setattr(
        m.ai_conversation_repository, "count_active_pinned", count_pinned or _default_count_pinned
    )
    monkeypatch.setattr(m.ai_conversation_repository, "set_pinned", set_pinned or _async)
    monkeypatch.setattr(
        m.ai_conversation_repository, "get_in_trash", get_in_trash or _default_get_in_trash
    )
    monkeypatch.setattr(m.ai_conversation_repository, "restore_by_ids", restore or _default_restore)
    monkeypatch.setattr(
        m.ai_conversation_repository, "paginate_trash", paginate_trash or _default_trash
    )
    monkeypatch.setattr(m.ai_conversation_repository, "get_by_ids", _default_get_by_ids)
    monkeypatch.setattr(m.ai_conversation_repository, "mark_read", mark_read or _async)
    monkeypatch.setattr(
        m.ai_message_repository, "get_last_message_id", get_last_msg or _default_last_msg
    )
    monkeypatch.setattr(
        m.ai_message_repository, "count_messages_after", count_after or _default_count_after
    )
    monkeypatch.setattr(m, "sync_conversation_to_es", _async)
    monkeypatch.setattr(m, "search_conversations", search or _default_search)
    if get_chain is not None:
        monkeypatch.setattr(m.ai_message_repository, "get_chain_by_id", get_chain)


class TestBatchOperate:
    async def test_archive_sets_status_2(self, monkeypatch):
        db = StubAsyncSession()
        updated = []

        async def upd(_db, ids, st):
            updated.append((ids, st))

        _patch(monkeypatch, update_status=upd)
        count = await ai_conversation_service.batch_operate(db, 1, "archive", [1, 2])
        assert count == 2
        assert updated == [([1], 2), ([2], 2)]

    async def test_batch_delete_requires_confirm(self, monkeypatch):
        db = StubAsyncSession()
        _patch(monkeypatch)
        with pytest.raises(BusinessException):
            await ai_conversation_service.batch_operate(db, 1, "delete", [1], confirm=False)

    async def test_batch_rollback_on_failure(self, monkeypatch):
        db = StubAsyncSession()
        calls = []

        async def upd(_db, ids, st):
            calls.append((ids, st))

        async def get_owned(_db, cid, uid):
            if cid == 2:
                raise BusinessException(m.ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
            return _conv(id=cid, status=1)

        _patch(monkeypatch, get_owned=get_owned, update_status=upd)
        with pytest.raises(BusinessException) as exc:
            await ai_conversation_service.batch_operate(db, 1, "archive", [1, 2])
        assert "会话不存在" in str(exc.value)
        assert calls == [([1], 2)]
        assert db.rolled_back == 1

    async def test_batch_restore_invalid_state_rolls_back(self, monkeypatch):
        db = StubAsyncSession()

        async def upd(_db, ids, st):
            pass

        async def get_owned(_db, cid, uid):
            if cid == 2:
                return _conv(id=cid, status=1)
            return _conv(id=cid, status=2)

        _patch(monkeypatch, get_owned=get_owned, update_status=upd)
        with pytest.raises(BusinessException):
            await ai_conversation_service.batch_operate(db, 1, "restore", [1, 2])
        assert db.rolled_back == 1


class TestPinLimit:
    async def test_pin_exceeds_limit_raises(self, monkeypatch):
        db = StubAsyncSession()

        async def count_pinned(_db, uid):
            return m.PINNED_CONVERSATION_LIMIT

        _patch(monkeypatch, count_pinned=count_pinned)
        with pytest.raises(BusinessException) as exc:
            await ai_conversation_service.pin_conversation(db, 1, 1)
        assert exc.value.code.code == "A0501"

    async def test_pin_sets_pinned_at(self, monkeypatch):
        db = StubAsyncSession()
        state = {}
        _patch(monkeypatch, set_pinned=_set_pin(state))
        result = await ai_conversation_service.pin_conversation(db, 1, 1)
        assert state["pinned"] == 1
        assert state["at"] is not None
        assert result.pinned == 1

    async def test_unpin_clears_pinned_at(self, monkeypatch):
        db = StubAsyncSession()
        state = {}
        _patch(monkeypatch, set_pinned=_set_pin(state))
        result = await ai_conversation_service.unpin_conversation(db, 1, 1)
        assert state["pinned"] == 0
        assert state["at"] is None
        assert result.pinned == 0


class TestRestoreWindow:
    async def test_restore_within_window(self, monkeypatch):
        db = StubAsyncSession()
        restored = []

        async def restore(_db, ids):
            restored.append(ids)

        async def in_trash(_db, cid, uid, win):
            return _conv(id=cid)

        _patch(monkeypatch, restore=restore, get_in_trash=in_trash)
        result = await ai_conversation_service.restore_conversation(db, 1, 1)
        assert restored == [[1]]
        assert result.id == 1

    async def test_restore_outside_window_raises(self, monkeypatch):
        db = StubAsyncSession()

        async def in_trash(_db, cid, uid, win):
            return None

        _patch(monkeypatch, get_in_trash=in_trash)
        with pytest.raises(BusinessException) as exc:
            await ai_conversation_service.restore_conversation(db, 1, 1)
        assert "恢复窗口" in str(exc.value)

    async def test_trash_passes_window(self, monkeypatch):
        db = StubAsyncSession()
        captured = {}

        async def paginate(_db, uid, p, s, win):
            captured["win"] = win
            return [_conv(id=1)], 1

        _patch(monkeypatch, paginate_trash=paginate)
        await ai_conversation_service.list_trash(db, 1, 1, 10)
        assert (datetime.now() - captured["win"]).days >= 29


class TestReadAndUnread:
    async def test_mark_read_sets_last_message_id(self, monkeypatch):
        db = StubAsyncSession()
        state = {}

        async def mark(_db, cid, mid):
            state["cid"] = cid
            state["mid"] = mid

        async def last_msg(_db, cid):
            return 10

        _patch(monkeypatch, mark_read=mark, get_last_msg=last_msg)
        await ai_conversation_service.mark_read(db, 1, 1)
        assert state == {"cid": 1, "mid": 10}

    async def test_unread_count_computed(self, monkeypatch):
        db = StubAsyncSession()

        async def count_after(_db, cid, aid):
            return 5

        _patch(monkeypatch, count_after=count_after)
        result = await ai_conversation_service._to_result(db, _conv(id=1, last_read_message_id=3))
        assert result.unread_count == 5

    async def test_unread_count_defaults_to_message_count(self):
        db = StubAsyncSession()
        result = await ai_conversation_service._to_result(
            db, _conv(id=1, last_read_message_id=None, message_count=7)
        )
        assert result.unread_count == 7


class TestExport:
    async def test_export_markdown_filters_non_dialogue(self, monkeypatch):
        db = StubAsyncSession()
        _patch(monkeypatch, get_chain=_export_chain)
        resp = await ai_conversation_service.export_conversation(db, 1, 1, "markdown")
        body = "".join([chunk async for chunk in resp.body_iterator])
        assert "# 测试会话" in body
        assert "## 用户" in body
        assert "## 助手" in body
        assert "帮我分析一下雾霾的形成原因" in body
        assert "工具返回：查询到监测数据" not in body
        assert resp.headers["Content-Disposition"].startswith(
            'attachment; filename="conversation_1.md"'
        )

    async def test_export_json_keeps_dialogue_messages_only(self, monkeypatch):
        db = StubAsyncSession()
        _patch(monkeypatch, get_chain=_export_chain)
        resp = await ai_conversation_service.export_conversation(db, 1, 1, "json")
        body = "".join([chunk async for chunk in resp.body_iterator])
        data = json.loads(body)
        assert data["conversation"]["title"] == "测试会话"
        roles = [msg["role"] for msg in data["messages"]]
        assert roles == ["user", "assistant", "assistant"]
        assert all(r in ("user", "assistant") for r in roles)
        assert resp.headers["Content-Disposition"].startswith(
            'attachment; filename="conversation_1.json"'
        )


class TestESList:
    async def test_es_search_passes_status_and_pagination(self, monkeypatch):
        db = StubAsyncSession()
        captured = {}

        async def search(user_id, query, *, status, page, size):
            captured.update(status=status, page=page, size=size)
            return [1, 2], 2

        _patch(monkeypatch, search=search)
        result = await ai_conversation_service.list_conversations(db, 1, 2, 5, keyword="雾", status=1)
        assert captured == {"status": 1, "page": 2, "size": 5}
        assert result.total == 2
        assert len(result.list) == 2

    async def test_es_search_defaults_active_status(self, monkeypatch):
        db = StubAsyncSession()
        captured = {}

        async def search(user_id, query, *, status, page, size):
            captured["status"] = status
            return [], 0

        _patch(monkeypatch, search=search)
        await ai_conversation_service.list_conversations(db, 1, 1, 10, keyword="x")
        assert captured["status"] == 1

    async def test_es_sort_pinned_first_then_time(self):
        now = datetime.now()
        convs = [
            _conv(id=1, pinned=0, pinned_at=None, last_message_at=now),
            _conv(id=2, pinned=1, pinned_at=now - timedelta(hours=2), last_message_at=None),
            _conv(id=3, pinned=1, pinned_at=now, last_message_at=None),
        ]
        result = ai_conversation_service._sort_conversations(convs)
        assert [c.id for c in result] == [3, 2, 1]

    async def test_keyword_es_empty_returns_empty_page(self, monkeypatch):
        """ES 必选：keyword 搜索 ES 无命中时直接返回空页，不降级 MySQL"""
        db = StubAsyncSession()
        called = {}
        _patch(monkeypatch, search=_es_empty, paginate_user_conversations=_paginate_capturing(called))
        result = await ai_conversation_service.list_conversations(db, 1, 1, 10, keyword="雾", status=2)
        assert result.list == [] and result.total == 0
        assert "status" not in called

    async def test_status_all_passes_none_to_es(self, monkeypatch):
        db = StubAsyncSession()
        es_status = {}

        async def search(user_id, query, *, status, page, size):
            es_status["status"] = status
            return [1], 1

        _patch(monkeypatch, search=search)
        await ai_conversation_service.list_conversations(db, 1, 1, 10, keyword="雾", status=0)
        assert es_status["status"] is None

    async def test_status_all_without_keyword_passes_none_to_mysql(self, monkeypatch):
        db = StubAsyncSession()
        captured = {}
        _patch(monkeypatch, paginate_user_conversations=_paginate_capturing(captured))
        await ai_conversation_service.list_conversations(db, 1, 1, 10, status=0)
        assert captured["status"] is None
