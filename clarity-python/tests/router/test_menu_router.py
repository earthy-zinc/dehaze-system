"""菜单管理路由层测试（权限装饰器 / 参数校验 / 越权路径）。

遵循 05-python-test-rules：marker=api、dependency_overrides + monkeypatch service、
只断言业务结果。权限口径：新增/修改/删除/改显示状态需
sys:menu:add|edit|delete（修改与显示状态共用 sys:menu:edit），
列表/表单/下拉/路由登录即可访问。
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.router import menu
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.api

MENU_PERMS = ["sys:menu:add", "sys:menu:edit", "sys:menu:delete"]


def _admin_ctx():
    return make_user_context(2, username="admin", roles=["ADMIN"], permissions=MENU_PERMS)


def _plain_ctx():
    """普通用户：无任何 sys:menu:* 权限"""
    return make_user_context(5, username="user", roles=[], permissions=[])


def _valid_form() -> dict:
    return {"parentId": 0, "name": "路由测试菜单", "type": 2, "path": "/router-test", "sort": 1}


@pytest.fixture
async def menu_client():
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


# ===== 写接口越权（T-MM-023/033/039/047：无 sys:menu:* 权限 → 403 A0301） =====


async def test_create_menu_without_permission_forbidden(menu_client):
    await _as_plain(menu_client)
    resp = await menu_client.post("/api/v1/menus", json=_valid_form())
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_update_menu_without_permission_forbidden(menu_client):
    await _as_plain(menu_client)
    resp = await menu_client.put("/api/v1/menus/1", json=_valid_form())
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_delete_menu_without_permission_forbidden(menu_client):
    await _as_plain(menu_client)
    resp = await menu_client.delete("/api/v1/menus/1")
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_update_menu_visible_without_permission_forbidden(menu_client):
    await _as_plain(menu_client)
    resp = await menu_client.patch("/api/v1/menus/1", json={"visible": 0})
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


# ===== 读接口登录即可访问（无 sys:menu:* 权限不拦，T-MM-001/054/058 前置） =====


async def test_read_endpoints_accessible_without_menu_permissions(menu_client, monkeypatch):
    await _as_plain(menu_client)

    async def fake_list_menus(db, keywords=None, perm=None, path=None, type=None, visible=None):
        return []

    async def fake_options(db):
        return []

    async def fake_routes(db, redis):
        return []

    async def fake_form(db, menu_id):
        return {"id": menu_id, "parentId": 0, "name": "表单菜单", "type": "MENU"}

    monkeypatch.setattr(menu.menu_service, "list_menus", fake_list_menus)
    monkeypatch.setattr(menu.menu_service, "list_menu_options", fake_options)
    monkeypatch.setattr(menu.menu_service, "list_routes", fake_routes)
    monkeypatch.setattr(menu.menu_service, "get_menu_form", fake_form)

    for url in (
        "/api/v1/menus",
        "/api/v1/menus/options",
        "/api/v1/menus/routes",
        "/api/v1/menus/1/form",
    ):
        resp = await menu_client.get(url)
        assert resp.status_code == 200, url
        assert resp.json()["code"] == "00000", url


# ===== 参数校验（Pydantic → A0400） =====


async def test_create_menu_rejects_invalid_type(menu_client):
    resp = await menu_client.post("/api/v1/menus", json={**_valid_form(), "type": 5})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_create_menu_rejects_negative_sort(menu_client):
    resp = await menu_client.post("/api/v1/menus", json={**_valid_form(), "sort": -1})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_create_menu_rejects_overlong_name(menu_client):
    resp = await menu_client.post("/api/v1/menus", json={**_valid_form(), "name": "m" * 65})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_update_visible_rejects_invalid_value(menu_client):
    resp = await menu_client.patch("/api/v1/menus/1", json={"visible": 2})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_update_visible_rejects_missing_value(menu_client):
    resp = await menu_client.patch("/api/v1/menus/1", json={})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


# ===== 查询参数边界 =====


async def test_list_menus_rejects_out_of_range_filters(menu_client):
    resp = await menu_client.get("/api/v1/menus", params={"type": 5})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"

    resp = await menu_client.get("/api/v1/menus", params={"visible": 2})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"
