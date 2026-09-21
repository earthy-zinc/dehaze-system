"""用户管理路由层测试（权限装饰器 / 参数校验 / 越权路径）。

遵循 05-python-test-rules：marker=api、构造注入（dependency_overrides + monkeypatch
service）、只断言业务结果（code/data），不测装饰器内部细节。
权限口径：新增/编辑/删除/重置密码需 sys:user:add|edit|delete|password:reset
（密码端点已收敛为纯管理员重置，本人也需权限，个人改密走 /api/v1/auth/password）；
状态接口需 sys:user:status。
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.router import user
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.api


def _admin_ctx():
    return make_user_context(
        2,
        username="admin",
        roles=["ADMIN"],
        permissions=[
            "sys:user:add",
            "sys:user:edit",
            "sys:user:delete",
            "sys:user:password:reset",
            "sys:user:status",
        ],
    )


def _plain_ctx():
    """普通用户：无任何 sys:user:* 权限"""
    return make_user_context(5, username="user", roles=[], permissions=[])


@pytest.fixture
async def user_client():
    async def _override_db():
        return object()

    async def _override_user():
        return _admin_ctx()

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as client:
        yield client
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)


async def _as_plain(client: AsyncClient):
    async def _override_user():
        return _plain_ctx()

    fastapi_app.dependency_overrides[get_current_user] = _override_user
    return client


def _valid_form() -> dict:
    return {
        "username": "tum_router_user",
        "nickname": "路由测试",
        "deptId": 1,
        "roleIds": [1],
    }


# ===== 写接口权限（403） =====


async def test_create_user_without_permission_forbidden(user_client):
    await _as_plain(user_client)
    resp = await user_client.post("/api/v1/users", json=_valid_form())
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_update_user_without_permission_forbidden(user_client):
    await _as_plain(user_client)
    resp = await user_client.put("/api/v1/users/8", json=_valid_form())
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_delete_user_without_permission_forbidden(user_client):
    await _as_plain(user_client)
    resp = await user_client.delete("/api/v1/users/8")
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_create_user_with_permission_ok(user_client, monkeypatch):
    async def fake_create(db, data):
        return None

    monkeypatch.setattr(user.user_service, "create_user_with_roles", fake_create)
    resp = await user_client.post("/api/v1/users", json=_valid_form())
    assert resp.status_code == 200
    assert resp.json()["code"] == "00000"


# ===== 密码接口（纯管理员重置，权限装饰器校验） =====


async def test_update_password_without_reset_permission_forbidden(user_client):
    """密码端点收敛为纯管理员重置：无 sys:user:password:reset（含改本人）→ 403 A0301"""
    await _as_plain(user_client)
    resp = await user_client.patch("/api/v1/users/5/password", json={"password": "Abcd1234"})
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_update_password_with_reset_permission_ok(user_client, monkeypatch):
    async def fake_update_password(db, redis, user_id, password):
        return None

    monkeypatch.setattr(user.user_service, "update_password", fake_update_password)
    resp = await user_client.patch("/api/v1/users/8/password", json={"password": "Abcd1234"})
    assert resp.status_code == 200
    assert resp.json()["code"] == "00000"


# ===== 状态接口 =====


async def test_update_status_passes_current_user_for_self_protection(user_client, monkeypatch):
    """路由必须透传 current_user（自禁保护依赖它）"""
    captured = {}

    async def fake_update_status(db, redis, user_id, status, current_user=None):
        captured["current_user"] = current_user
        captured["user_id"] = user_id
        captured["status"] = status

    monkeypatch.setattr(user.user_service, "update_user_status", fake_update_status)
    resp = await user_client.patch("/api/v1/users/8/status", params={"status": 0})
    assert resp.status_code == 200
    assert captured["user_id"] == 8
    assert captured["status"] == 0
    assert captured["current_user"] is not None
    assert captured["current_user"].id == 2


async def test_update_status_without_status_permission_forbidden(user_client):
    """文档契约：状态接口需 sys:user:status 权限（权限码取自 API接口.md 权限矩阵，
    种子见 sys_menu.sql id=167，ADMIN 角色经 sys_role_menu 授权）"""
    await _as_plain(user_client)
    resp = await user_client.patch("/api/v1/users/8/status", params={"status": 0})
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


# ===== 分页参数校验 =====


async def test_get_page_page_num_zero_rejected(user_client):
    resp = await user_client.get("/api/v1/users/page", params={"pageNum": 0})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_get_page_page_size_over_limit_rejected(user_client):
    resp = await user_client.get("/api/v1/users/page", params={"pageSize": 101})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_get_page_status_out_of_range_rejected(user_client):
    resp = await user_client.get("/api/v1/users/page", params={"status": 2})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_get_page_xss_keywords_no_error(user_client, monkeypatch):
    """对抗性语料：XSS 关键词搜索正常返回（文档 T-UM-008，无 500/注入）"""

    async def fake_get_list(db, **kwargs):
        return [], 0

    monkeypatch.setattr(user.user_service, "get_user_list", fake_get_list)
    resp = await user_client.get(
        "/api/v1/users/page", params={"keywords": "<script>alert(1)</script>"}
    )
    assert resp.status_code == 200
    assert resp.json()["data"] == {"list": [], "total": 0}
