"""消息通知路由层测试（参数校验 / 登录约束 / 通知设置格式校验）。

遵循 05-python-test-rules：marker=api、dependency_overrides + monkeypatch service。
消息模块所有接口无独立权限标识（登录即可），写路径越权由服务层归属校验兜底
（见 tests/service/test_message_service.py）。
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.api


def _user_ctx(user_id: int = 1):
    """持 message:send 权限的用户（send 参数校验用例需先过权限层）"""
    return make_user_context(
        user_id, username="user", roles=["ADMIN"], permissions=["message:send"]
    )


@pytest.fixture
async def client():
    async def _override_db():
        return object()

    async def _override_user():
        return _user_ctx()

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as c:
        yield c
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)


async def _as_plain(client: AsyncClient):
    """切换为无任何权限的普通用户"""

    async def _override_user():
        return make_user_context(5, username="plain", roles=[], permissions=[])

    fastapi_app.dependency_overrides[get_current_user] = _override_user


# ===== 删除参数校验（A0400，对齐 Go parseIDsFromCSV 契约） =====


async def test_delete_invalid_ids_returns_param_error(client):
    resp = await client.delete("/api/v1/messages/abc,def")
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_delete_blank_ids_returns_param_error(client):
    resp = await client.delete("/api/v1/messages/,,")
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


# ===== 发送参数校验 =====


async def test_send_without_permission_forbidden(client):
    """普通用户（无 message:send）调内部发送接口被拒（A0301），防止伪造系统消息"""
    await _as_plain(client)
    resp = await client.post(
        "/api/v1/messages/send",
        json={"type": "business", "title": "t", "content": "c", "recipientIds": [1]},
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_send_with_permission_reaches_service(client, monkeypatch):
    """持 message:send 权限的用户通过权限层进入服务"""
    from app.service import message_service as svc_mod

    captured = {}

    async def _fake_send(db, data):
        captured["data"] = data
        return [123]

    monkeypatch.setattr(svc_mod.message_service, "send", _fake_send)
    resp = await client.post(
        "/api/v1/messages/send",
        json={"type": "business", "title": "t", "content": "c", "recipientIds": [1]},
    )
    assert resp.status_code == 200
    assert resp.json()["data"] == {"messageIds": [123]}


async def test_send_missing_type_rejected(client):
    resp = await client.post(
        "/api/v1/messages/send",
        json={"recipientIds": [1], "title": "t", "content": "c"},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_send_empty_recipients_rejected(client):
    resp = await client.post(
        "/api/v1/messages/send",
        json={"type": "business", "title": "t", "content": "c", "recipientIds": []},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_send_invalid_priority_rejected(client):
    resp = await client.post(
        "/api/v1/messages/send",
        json={
            "type": "business",
            "title": "t",
            "content": "c",
            "recipientIds": [1],
            "priority": 9,
        },
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


# ===== 列表分页边界 =====


async def test_get_page_rejects_oversized_page_size(client):
    resp = await client.get("/api/v1/messages", params={"pageNum": 1, "pageSize": 101})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_get_page_rejects_zero_page_num(client):
    resp = await client.get("/api/v1/messages", params={"pageNum": 0})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_search_requires_keyword(client):
    resp = await client.get("/api/v1/messages/search")
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


# ===== 通知设置格式校验 =====


async def test_update_settings_invalid_dnd_time_rejected(client):
    resp = await client.patch(
        "/api/v1/notification-settings",
        json={"dndStart": "25:99:00"},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_update_settings_valid_dnd_time_accepted(client, monkeypatch):
    """合法时间格式通过 schema 校验进入服务层（服务层已 monkeypatch，仅验路由）"""
    from app.service import notification_setting_service as svc_mod

    captured = {}

    async def _fake_update(db, user_id, data):
        captured["data"] = data

    monkeypatch.setattr(svc_mod.notification_setting_service, "update", _fake_update)
    resp = await client.patch(
        "/api/v1/notification-settings",
        json={"dndStart": "22:30:00", "dndEnd": "07:00:00"},
    )
    assert resp.status_code == 200
    assert captured["data"]["dndStart"] == "22:30:00"
