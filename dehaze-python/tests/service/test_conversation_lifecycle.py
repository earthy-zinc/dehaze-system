import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.requires_db

from app.core.exceptions import BusinessException
from app.service import ai_conversation_service as m
from app.service.ai_conversation_service import AiConversationService


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

    async def get_by_ids(self, db, uid, ids):
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
        ai_conversation_repository=conv_repo,
        ai_message_repository=msg_repo,
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
        state = {}
        svc = _make_service(conv={"set_pinned": _set_pin(state)})
        result = await svc.pin_conversation(db, 1, 1)
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
        result = await svc._to_result(
            db, _conv(id=1, last_read_message_id=None, message_count=7)
        )
        assert result.unread_count == 7


class TestExport:
    async def test_export_markdown_filters_non_dialogue(self, db):
        svc = _make_service(msg={"get_chain_by_id": _export_chain})
        resp = await svc.export_conversation(db, 1, 1, "markdown")
        body = "".join([chunk async for chunk in resp.body_iterator])
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
        body = "".join([chunk async for chunk in resp.body_iterator])
        data = json.loads(body)
        assert data["conversation"]["title"] == "测试会话"
        roles = [msg["role"] for msg in data["messages"]]
        assert roles == ["user", "assistant", "assistant"]
        assert all(r in ("user", "assistant") for r in roles)
        assert resp.headers["Content-Disposition"].startswith(
            'attachment; filename="conversation_1.json"'
        )


class TestESSyncDefer:
    """ES 读模型同步延迟到事务提交后：Service 层只登记 db.info，不直接触 ES"""

    async def test_restore_registers_post_commit_sync(self, db):
        svc = _make_service()
        await svc.restore_conversation(db, 1, 1)
        assert db.info.get("es_sync_conv_ids") == {1}

    async def test_batch_archive_registers_sync_for_each(self, db):
        svc = _make_service()
        await svc.batch_operate(db, 1, "archive", [1, 2])
        assert db.info.get("es_sync_conv_ids") == {1, 2}


class TestESList:
    async def test_es_search_passes_status_and_pagination(self, db, monkeypatch):
        captured = {}

        async def search(user_id, query, *, status, page, size):
            captured.update(status=status, page=page, size=size)
            return [1, 2], 2

        monkeypatch.setattr(m, "search_conversations", search)
        svc = _make_service(conv={"get_by_ids": _default_get_by_ids})
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
        svc = _make_service(
            conv={"paginate_user_conversations": _paginate_capturing(called)}
        )
        result = await svc.list_conversations(db, 1, 1, 10, keyword="雾", status=2)
        assert result.list == [] and result.total == 0
        assert "status" not in called

    async def test_status_all_passes_none_to_es(self, db, monkeypatch):
        es_status = {}

        async def search(user_id, query, *, status, page, size):
            es_status["status"] = status
            return [1], 1

        monkeypatch.setattr(m, "search_conversations", search)
        svc = _make_service(conv={"get_by_ids": _default_get_by_ids})
        await svc.list_conversations(db, 1, 1, 10, keyword="雾", status=0)
        assert es_status["status"] is None

    async def test_status_all_without_keyword_passes_none_to_mysql(self, db):
        captured = {}
        svc = _make_service(
            conv={"paginate_user_conversations": _paginate_capturing(captured)}
        )
        await svc.list_conversations(db, 1, 1, 10, status=0)
        assert captured["status"] is None


async def _default_get_by_ids(db, uid, ids):
    return [_conv(id=i) for i in ids]
