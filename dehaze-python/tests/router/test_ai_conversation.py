"""AI 会话路由测试：CRUD / 审计视角(view=admin) / 回收站 / 批量操作 / 置顶 / 导出

覆盖重点：
- 路径注册与参数校验（A0400）
- view=admin 审计视角权限拦截（A0301）与审计字段 camelCase 输出
- 置顶、回收站恢复、批量操作（含 confirm 透传）
- 业务错误码透传（A0401/A0502/A0501）
"""
from datetime import datetime

import pytest
from fastapi.responses import StreamingResponse
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.api

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.models.schema.ai_conversation import ConversationResult
from app.models.schema.common import PageResult
from app.service.ai_conversation_service import ai_conversation_service

AUDIT_PERM = "ai:conversation:audit"


class _FakeUser:
    def __init__(self, id=1, is_root=False, permissions=()):
        self.id = id
        self.is_root = is_root
        self.permissions = list(permissions)


def _conversation(**overrides) -> ConversationResult:
    base = {
        "id": 1,
        "user_id": 1,
        "title": "测试会话",
        "agent_code": "default",
        "message_count": 2,
        "pinned": 0,
        "title_source": "manual",
        "status": 1,
        "create_time": datetime(2026, 8, 29, 10, 0, 0),
    }
    base.update(overrides)
    return ConversationResult.model_validate(base)


@pytest.fixture
async def conv_client():
    async def _override_db():
        return object()

    current_user = {"user": _FakeUser()}

    async def _override_user():
        return current_user["user"]

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as client:
        yield client, current_user
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)


def test_conversation_paths_registered(app):
    schema = app.openapi()
    for path in (
        "/api/v1/ai/conversations",
        "/api/v1/ai/conversations/trash",
        "/api/v1/ai/conversations/batch",
        "/api/v1/ai/conversations/{conv_id}",
        "/api/v1/ai/conversations/{conv_id}/restore",
        "/api/v1/ai/conversations/{conv_id}/pin",
        "/api/v1/ai/conversations/{conv_id}/unpin",
        "/api/v1/ai/conversations/{conv_id}/read",
        "/api/v1/ai/conversations/{conv_id}/export",
        "/api/v1/ai/conversations/{conv_id}/messages",
    ):
        assert path in schema["paths"], f"缺少路径 {path}"


class TestCreate:
    async def test_create_success(self, conv_client, monkeypatch):
        client, state = conv_client
        state["user"] = _FakeUser(id=7)
        captured: dict = {}

        async def _fake_create(db, user_id, form):
            captured.update(user_id=user_id, form=form)
            return _conversation(id=11, title=form.title or "新对话")

        monkeypatch.setattr(ai_conversation_service, "create_conversation", _fake_create)
        resp = await client.post(
            "/api/v1/ai/conversations",
            json={"title": "我的会话", "agentCode": "dehaze", "scene": "multi_step"},
        )
        assert resp.status_code == 200
        assert captured["user_id"] == 7
        assert captured["form"].agentCode == "dehaze"
        assert captured["form"].scene == "multi_step"
        data = resp.json()["data"]
        assert data["id"] == 11
        assert data["title"] == "我的会话"
        assert data["agentCode"] == "default"
        assert data["messageCount"] == 2

    async def test_create_title_too_long_rejected(self, conv_client):
        client, _ = conv_client
        resp = await client.post("/api/v1/ai/conversations", json={"title": "x" * 256})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_create_model_too_long_rejected(self, conv_client):
        client, _ = conv_client
        resp = await client.post("/api/v1/ai/conversations", json={"model": "m" * 65})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"


class TestList:
    async def test_list_default_view_forwards_user_scope(self, conv_client, monkeypatch):
        client, state = conv_client
        state["user"] = _FakeUser(id=3)
        captured: dict = {}

        async def _fake_list(db, user_id, page, size, keyword=None, status=None, view=None):
            captured.update(
                user_id=user_id, page=page, size=size, keyword=keyword, status=status, view=view
            )
            return PageResult(list=[_conversation()], total=1)

        monkeypatch.setattr(ai_conversation_service, "list_conversations", _fake_list)
        resp = await client.get(
            "/api/v1/ai/conversations",
            params={"keyword": "abc", "status": 1, "pageNum": 2, "pageSize": 5},
        )
        assert resp.status_code == 200
        assert captured == {
            "user_id": 3,
            "page": 2,
            "size": 5,
            "keyword": "abc",
            "status": 1,
            "view": None,
        }
        assert resp.json()["data"]["total"] == 1
        assert resp.json()["data"]["list"][0]["title"] == "测试会话"

    async def test_list_admin_view_forbidden_without_permission(self, conv_client, monkeypatch):
        client, state = conv_client
        state["user"] = _FakeUser(id=3, permissions=[])

        async def _fake_list(db, user_id, page, size, keyword=None, status=None, view=None):
            return PageResult(list=[], total=0)

        monkeypatch.setattr(ai_conversation_service, "list_conversations", _fake_list)
        resp = await client.get("/api/v1/ai/conversations", params={"view": "admin"})
        assert resp.status_code == 403
        assert resp.json()["code"] == "A0301"

    async def test_list_admin_view_allowed_with_permission(self, conv_client, monkeypatch):
        client, state = conv_client
        state["user"] = _FakeUser(id=3, permissions=[AUDIT_PERM])
        captured: dict = {}

        async def _fake_list(db, user_id, page, size, keyword=None, status=None, view=None):
            captured.update(user_id=user_id, view=view)
            return PageResult(
                list=[
                    _conversation(
                        id=9,
                        user_id=42,
                        user_name="张三",
                        token_consumed=1200,
                        credits_consumed=30,
                        anomaly_type="failed",
                        anomaly_label="存在失败消息",
                    )
                ],
                total=1,
            )

        monkeypatch.setattr(ai_conversation_service, "list_conversations", _fake_list)
        resp = await client.get("/api/v1/ai/conversations", params={"view": "admin"})
        assert resp.status_code == 200
        assert captured["view"] == "admin"
        item = resp.json()["data"]["list"][0]
        assert item["userId"] == 42
        assert item["userName"] == "张三"
        assert item["tokenConsumed"] == 1200
        assert item["creditsConsumed"] == 30
        assert item["anomalyType"] == "failed"
        assert item["anomalyLabel"] == "存在失败消息"

    async def test_list_admin_view_allowed_for_root(self, conv_client, monkeypatch):
        client, state = conv_client
        state["user"] = _FakeUser(id=3, is_root=True)
        captured: dict = {}

        async def _fake_list(db, user_id, page, size, keyword=None, status=None, view=None):
            captured["view"] = view
            return PageResult(list=[], total=0)

        monkeypatch.setattr(ai_conversation_service, "list_conversations", _fake_list)
        resp = await client.get("/api/v1/ai/conversations", params={"view": "admin"})
        assert resp.status_code == 200
        assert captured["view"] == "admin"

    async def test_list_invalid_page_num_rejected(self, conv_client):
        client, _ = conv_client
        resp = await client.get("/api/v1/ai/conversations", params={"pageNum": 0})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"


class TestDetail:
    async def test_detail_owner_view_not_admin(self, conv_client, monkeypatch):
        client, state = conv_client
        state["user"] = _FakeUser(id=5, permissions=[])
        captured: dict = {}

        async def _fake_get(db, conv_id, user_id, admin=False):
            captured.update(conv_id=conv_id, user_id=user_id, admin=admin)
            return _conversation(id=conv_id, user_id=user_id)

        monkeypatch.setattr(ai_conversation_service, "get_conversation", _fake_get)
        resp = await client.get("/api/v1/ai/conversations/8")
        assert resp.status_code == 200
        assert captured == {"conv_id": 8, "user_id": 5, "admin": False}

    async def test_detail_admin_view_requires_permission(self, conv_client):
        client, state = conv_client
        state["user"] = _FakeUser(id=5, permissions=[])
        resp = await client.get("/api/v1/ai/conversations/8", params={"view": "admin"})
        assert resp.status_code == 403
        assert resp.json()["code"] == "A0301"

    async def test_detail_admin_view_with_audit_fields(self, conv_client, monkeypatch):
        client, state = conv_client
        state["user"] = _FakeUser(id=5, permissions=[AUDIT_PERM])
        captured: dict = {}

        async def _fake_get(db, conv_id, user_id, admin=False):
            captured.update(admin=admin)
            return _conversation(id=conv_id, user_id=99, user_name="李四", token_consumed=0)

        monkeypatch.setattr(ai_conversation_service, "get_conversation", _fake_get)
        resp = await client.get("/api/v1/ai/conversations/8", params={"view": "admin"})
        assert resp.status_code == 200
        assert captured["admin"] is True
        assert resp.json()["data"]["userName"] == "李四"

    async def test_detail_not_found_maps_a0401(self, conv_client, monkeypatch):
        client, _ = conv_client

        async def _fake_get(db, conv_id, user_id, admin=False):
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")

        monkeypatch.setattr(ai_conversation_service, "get_conversation", _fake_get)
        resp = await client.get("/api/v1/ai/conversations/404")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0401"


class TestUpdate:
    async def test_update_forwards_form(self, conv_client, monkeypatch):
        client, state = conv_client
        state["user"] = _FakeUser(id=6)
        captured: dict = {}

        async def _fake_update(db, conv_id, user_id, form):
            captured.update(conv_id=conv_id, user_id=user_id, form=form)
            return _conversation(id=conv_id, title=form.title or "测试会话", pinned=1)

        monkeypatch.setattr(ai_conversation_service, "update_conversation", _fake_update)
        resp = await client.patch(
            "/api/v1/ai/conversations/12", json={"title": "改后标题", "pinned": 1}
        )
        assert resp.status_code == 200
        assert captured["conv_id"] == 12
        assert captured["form"].title == "改后标题"
        assert resp.json()["data"]["pinned"] == 1

    async def test_update_pin_limit_exceeded_maps_a0501(self, conv_client, monkeypatch):
        client, _ = conv_client

        async def _fake_update(db, conv_id, user_id, form):
            raise BusinessException(ResultCode.DATA_EXISTS, "置顶会话已达上限")

        monkeypatch.setattr(ai_conversation_service, "update_conversation", _fake_update)
        resp = await client.patch("/api/v1/ai/conversations/12", json={"pinned": 1})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0501"

    async def test_update_invalid_status_rejected(self, conv_client):
        client, _ = conv_client
        resp = await client.patch("/api/v1/ai/conversations/12", json={"agentCode": "x" * 65})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"


class TestLifecycle:
    async def test_soft_delete(self, conv_client, monkeypatch):
        client, state = conv_client
        state["user"] = _FakeUser(id=4)
        captured: dict = {}

        async def _fake_delete(db, conv_id, user_id):
            captured.update(conv_id=conv_id, user_id=user_id)

        monkeypatch.setattr(ai_conversation_service, "delete_conversation", _fake_delete)
        resp = await client.delete("/api/v1/ai/conversations/21")
        assert resp.status_code == 200
        assert captured == {"conv_id": 21, "user_id": 4}

    async def test_restore(self, conv_client, monkeypatch):
        client, _ = conv_client

        async def _fake_restore(db, conv_id, user_id):
            return _conversation(id=conv_id, user_id=user_id)

        monkeypatch.setattr(ai_conversation_service, "restore_conversation", _fake_restore)
        resp = await client.post("/api/v1/ai/conversations/21/restore")
        assert resp.status_code == 200
        assert resp.json()["data"]["id"] == 21

    async def test_restore_out_of_window_maps_a0401(self, conv_client, monkeypatch):
        client, _ = conv_client

        async def _fake_restore(db, conv_id, user_id):
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在或已超出恢复窗口")

        monkeypatch.setattr(ai_conversation_service, "restore_conversation", _fake_restore)
        resp = await client.post("/api/v1/ai/conversations/21/restore")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0401"

    async def test_trash_list_forwards_paging(self, conv_client, monkeypatch):
        client, state = conv_client
        state["user"] = _FakeUser(id=4)
        captured: dict = {}

        async def _fake_trash(db, user_id, page, size):
            captured.update(user_id=user_id, page=page, size=size)
            return PageResult(list=[], total=0)

        monkeypatch.setattr(ai_conversation_service, "list_trash", _fake_trash)
        resp = await client.get(
            "/api/v1/ai/conversations/trash", params={"pageNum": 3, "pageSize": 20}
        )
        assert resp.status_code == 200
        assert captured == {"user_id": 4, "page": 3, "size": 20}


class TestPinAndRead:
    async def test_pin(self, conv_client, monkeypatch):
        client, state = conv_client
        state["user"] = _FakeUser(id=4)
        captured: dict = {}

        async def _fake_pin(db, conv_id, user_id):
            captured.update(conv_id=conv_id, user_id=user_id)
            return _conversation(id=conv_id, pinned=1, pinned_at=datetime(2026, 8, 29, 11, 0, 0))

        monkeypatch.setattr(ai_conversation_service, "pin_conversation", _fake_pin)
        resp = await client.put("/api/v1/ai/conversations/31/pin")
        assert resp.status_code == 200
        assert captured == {"conv_id": 31, "user_id": 4}
        assert resp.json()["data"]["pinned"] == 1

    async def test_pin_limit_maps_a0501(self, conv_client, monkeypatch):
        client, _ = conv_client

        async def _fake_pin(db, conv_id, user_id):
            raise BusinessException(ResultCode.DATA_EXISTS, "置顶会话已达上限")

        monkeypatch.setattr(ai_conversation_service, "pin_conversation", _fake_pin)
        resp = await client.put("/api/v1/ai/conversations/31/pin")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0501"

    async def test_unpin(self, conv_client, monkeypatch):
        client, _ = conv_client

        async def _fake_unpin(db, conv_id, user_id):
            return _conversation(id=conv_id, pinned=0)

        monkeypatch.setattr(ai_conversation_service, "unpin_conversation", _fake_unpin)
        resp = await client.put("/api/v1/ai/conversations/31/unpin")
        assert resp.status_code == 200
        assert resp.json()["data"]["pinned"] == 0

    async def test_mark_read(self, conv_client, monkeypatch):
        client, state = conv_client
        state["user"] = _FakeUser(id=4)
        captured: dict = {}

        async def _fake_read(db, conv_id, user_id):
            captured.update(conv_id=conv_id, user_id=user_id)
            return _conversation(id=conv_id, last_read_message_id=100, unread_count=0)

        monkeypatch.setattr(ai_conversation_service, "mark_read", _fake_read)
        resp = await client.put("/api/v1/ai/conversations/31/read")
        assert resp.status_code == 200
        assert captured == {"conv_id": 31, "user_id": 4}
        assert resp.json()["data"]["lastReadMessageId"] == 100
        assert resp.json()["data"]["unreadCount"] == 0


class TestBatch:
    async def test_batch_delete_forwards_confirm(self, conv_client, monkeypatch):
        client, state = conv_client
        state["user"] = _FakeUser(id=4)
        captured: dict = {}

        async def _fake_batch(db, user_id, action, ids, confirm):
            captured.update(user_id=user_id, action=action, ids=ids, confirm=confirm)
            return len(ids)

        monkeypatch.setattr(ai_conversation_service, "batch_operate", _fake_batch)
        resp = await client.post(
            "/api/v1/ai/conversations/batch",
            json={"action": "delete", "ids": [1, 2, 3], "confirm": True},
        )
        assert resp.status_code == 200
        assert captured == {
            "user_id": 4,
            "action": "delete",
            "ids": [1, 2, 3],
            "confirm": True,
        }
        assert resp.json()["data"] == 3

    async def test_batch_archive_without_confirm(self, conv_client, monkeypatch):
        client, _ = conv_client
        captured: dict = {}

        async def _fake_batch(db, user_id, action, ids, confirm):
            captured.update(action=action, confirm=confirm)
            return 1

        monkeypatch.setattr(ai_conversation_service, "batch_operate", _fake_batch)
        resp = await client.post(
            "/api/v1/ai/conversations/batch", json={"action": "archive", "ids": [5]}
        )
        assert resp.status_code == 200
        assert captured == {"action": "archive", "confirm": False}

    async def test_batch_state_not_allowed_maps_a0502(self, conv_client, monkeypatch):
        client, _ = conv_client

        async def _fake_batch(db, user_id, action, ids, confirm):
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "仅活跃会话可归档")

        monkeypatch.setattr(ai_conversation_service, "batch_operate", _fake_batch)
        resp = await client.post(
            "/api/v1/ai/conversations/batch", json={"action": "archive", "ids": [5]}
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0502"

    async def test_batch_invalid_action_rejected(self, conv_client):
        client, _ = conv_client
        resp = await client.post(
            "/api/v1/ai/conversations/batch", json={"action": "unknown", "ids": [5]}
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_batch_empty_ids_rejected(self, conv_client):
        client, _ = conv_client
        resp = await client.post(
            "/api/v1/ai/conversations/batch", json={"action": "delete", "ids": []}
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"


class TestExport:
    async def test_export_markdown_stream(self, conv_client, monkeypatch):
        client, _ = conv_client
        captured: dict = {}

        async def _fake_export(db, conv_id, user_id, fmt):
            captured.update(conv_id=conv_id, user_id=user_id, fmt=fmt)
            return StreamingResponse(
                iter(["# 测试会话\n\n用户：你好\n"]),
                media_type="text/markdown; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="conversation_1.md"'},
            )

        monkeypatch.setattr(ai_conversation_service, "export_conversation", _fake_export)
        resp = await client.get("/api/v1/ai/conversations/1/export")
        assert resp.status_code == 200
        assert captured == {"conv_id": 1, "user_id": 1, "fmt": "markdown"}
        assert resp.headers["content-type"].startswith("text/markdown")
        assert "conversation_1.md" in resp.headers["content-disposition"]
        assert "# 测试会话" in resp.text

    async def test_export_json_stream(self, conv_client, monkeypatch):
        client, _ = conv_client
        captured: dict = {}

        async def _fake_export(db, conv_id, user_id, fmt):
            captured["fmt"] = fmt
            return StreamingResponse(
                iter(['{"conversation": {"id": 1}}']),
                media_type="application/json; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="conversation_1.json"'},
            )

        monkeypatch.setattr(ai_conversation_service, "export_conversation", _fake_export)
        resp = await client.get("/api/v1/ai/conversations/1/export", params={"format": "json"})
        assert resp.status_code == 200
        assert captured["fmt"] == "json"
        assert resp.json()["conversation"]["id"] == 1

    async def test_export_invalid_format_rejected(self, conv_client):
        client, _ = conv_client
        resp = await client.get("/api/v1/ai/conversations/1/export", params={"format": "pdf"})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"
