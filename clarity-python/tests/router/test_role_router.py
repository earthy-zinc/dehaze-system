"""角色管理路由层测试（权限装饰器 / 参数校验 / 越权路径）。

遵循 05-python-test-rules：marker=api、dependency_overrides + monkeypatch service、
只断言业务结果。权限口径：新增/删除/改状态/分配菜单需 sys:role:add|edit|delete
（编辑与状态/分配共用 sys:role:edit），page/options/form/menuIds 登录即可访问。
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.router import role
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.api

ROLE_PERMS = ["sys:role:add", "sys:role:edit", "sys:role:delete"]


def _admin_ctx():
    return make_user_context(2, username="admin", roles=["ADMIN"], permissions=ROLE_PERMS)


def _plain_ctx():
    """普通用户：无任何 sys:role:* 权限"""
    return make_user_context(5, username="user", roles=[], permissions=[])


def _valid_form() -> dict:
    return {
        "name": "路由测试角色",
        "code": "ROUTER_TEST_ROLE",
        "dataScope": 1,
        "sort": 1,
        "status": 1,
    }


@pytest.fixture
async def role_client():
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


# ===== 写接口越权（403 A0301，T-RM-014/020/027 + 状态/分配菜单越权） =====


async def test_create_role_without_permission_forbidden(role_client):
    await _as_plain(role_client)
    resp = await role_client.post("/api/v1/roles", json=_valid_form())
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_update_role_without_permission_forbidden(role_client):
    await _as_plain(role_client)
    resp = await role_client.put("/api/v1/roles/2", json=_valid_form())
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_delete_role_without_permission_forbidden(role_client):
    await _as_plain(role_client)
    resp = await role_client.delete("/api/v1/roles/2")
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_update_role_status_without_permission_forbidden(role_client):
    await _as_plain(role_client)
    resp = await role_client.patch("/api/v1/roles/2/status", params={"status": 0})
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_assign_menus_without_permission_forbidden(role_client):
    await _as_plain(role_client)
    resp = await role_client.patch("/api/v1/roles/2/menus", json=[1, 2])
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


# ===== 读接口登录即可访问（无 sys:role:* 权限不拦） =====


async def test_read_endpoints_accessible_without_role_permissions(role_client, monkeypatch):
    await _as_plain(role_client)

    async def fake_list(db, page, page_size, keywords=None):
        return [], 0

    async def fake_options(db, redis, *, is_root=False):
        return []

    async def fake_get_by_id(db, role_id):
        return None

    async def fake_menu_ids(db, role_id):
        return []

    monkeypatch.setattr(role.role_service, "get_role_list", fake_list)
    monkeypatch.setattr(role.role_service, "get_role_options", fake_options)
    monkeypatch.setattr(role.role_service, "get_role_by_id", fake_get_by_id)
    monkeypatch.setattr(role.role_service, "get_role_menu_ids", fake_menu_ids)

    for path in (
        "/api/v1/roles/page",
        "/api/v1/roles/options",
        "/api/v1/roles/2/form",
        "/api/v1/roles/2/menuIds",
    ):
        resp = await role_client.get(path)
        assert resp.status_code == 200
        assert resp.json()["code"] == "00000"


# ===== 有权限路径放行（service 打桩，验证权限装饰器与参数解析） =====


async def test_create_role_with_permission_ok(role_client, monkeypatch):
    async def fake_create(db, redis, data):
        return None

    monkeypatch.setattr(role.role_service, "create_role", fake_create)
    resp = await role_client.post("/api/v1/roles", json=_valid_form())
    assert resp.status_code == 200
    assert resp.json()["code"] == "00000"


async def test_assign_menus_with_permission_ok(role_client, monkeypatch):
    captured = {}

    async def fake_assign(db, redis, role_id, menu_ids, *, operator):
        captured["role_id"] = role_id
        captured["menu_ids"] = menu_ids
        captured["operator"] = operator

    monkeypatch.setattr(role.role_service, "assign_menus_to_role", fake_assign)
    resp = await role_client.patch("/api/v1/roles/2/menus", json=[1, 2, 3])
    assert resp.status_code == 200
    assert resp.json()["code"] == "00000"
    assert captured["role_id"] == 2
    assert captured["menu_ids"] == [1, 2, 3]
    assert captured["operator"].permissions == ROLE_PERMS


# ===== 参数校验（A0400，schema 层拦截） =====


async def test_create_role_data_scope_out_of_range_rejected(role_client):
    resp = await role_client.post("/api/v1/roles", json={**_valid_form(), "dataScope": 99})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_create_role_negative_sort_rejected(role_client):
    resp = await role_client.post("/api/v1/roles", json={**_valid_form(), "sort": -1})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_create_role_invalid_status_rejected(role_client):
    resp = await role_client.post("/api/v1/roles", json={**_valid_form(), "status": 2})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_create_role_missing_name_rejected(role_client):
    form = {k: v for k, v in _valid_form().items() if k != "name"}
    resp = await role_client.post("/api/v1/roles", json=form)
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_create_role_xss_name_rejected(role_client):
    resp = await role_client.post(
        "/api/v1/roles", json={**_valid_form(), "name": "<script>alert(1)</script>"}
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_update_role_status_out_of_range_rejected(role_client):
    resp = await role_client.patch("/api/v1/roles/2/status", params={"status": 2})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_role_page_invalid_page_size_rejected(role_client):
    resp = await role_client.get("/api/v1/roles/page", params={"pageNum": 1, "pageSize": 0})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"
