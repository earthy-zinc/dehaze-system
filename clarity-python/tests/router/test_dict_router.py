"""字典管理路由层测试（权限装饰器 / 参数校验 / 只读约束信封）。

对应字典管理测试用例.md：T-DM-013/018/026/041/048/052（六写接口越权 403 A0301）、
读接口登录即可访问（§3 权限标识汇总）、T-DM-006（无效分页）、删除非法 ID A0400、
typeCode 必填（API接口.md §2.2）、XSS 输入拦截（§7 安全性）。
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.dependencies.redis import get_redis
from app.main import app as fastapi_app
from app.router import dict as dict_router
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.api

ALL_PERMS = [
    "sys:dict:type:add",
    "sys:dict:type:edit",
    "sys:dict:type:delete",
    "sys:dict:data:add",
    "sys:dict:data:edit",
    "sys:dict:data:delete",
]


def _admin_ctx():
    return make_user_context(2, username="admin", roles=["ADMIN"], permissions=ALL_PERMS)


def _plain_ctx():
    """普通用户：无任何 sys:dict:* 权限"""
    return make_user_context(5, username="user", roles=[], permissions=[])


@pytest.fixture
async def dict_client():
    async def _override_stub():
        return object()

    fastapi_app.dependency_overrides[get_db] = _override_stub
    fastapi_app.dependency_overrides[get_redis] = _override_stub
    fastapi_app.dependency_overrides[get_current_user] = _admin_ctx
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as client:
        yield client
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_redis, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)


async def _as_plain():
    fastapi_app.dependency_overrides[get_current_user] = _plain_ctx


_TYPE_FORM = {"name": "路由测试类型", "code": "ROUTER_TEST_TYPE", "status": 1}
_DICT_FORM = {"typeCode": "gender", "name": "路由测试字典", "value": "999", "sort": 1, "status": 1}


# ===== 写接口越权（403 A0301） =====


async def test_create_dict_type_without_permission_forbidden(dict_client):
    await _as_plain()
    resp = await dict_client.post("/api/v1/dict/types", json=_TYPE_FORM)
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_update_dict_type_without_permission_forbidden(dict_client):
    await _as_plain()
    resp = await dict_client.put("/api/v1/dict/types/1", json=_TYPE_FORM)
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_delete_dict_types_without_permission_forbidden(dict_client):
    await _as_plain()
    resp = await dict_client.delete("/api/v1/dict/types/1")
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_create_dict_without_permission_forbidden(dict_client):
    await _as_plain()
    resp = await dict_client.post("/api/v1/dict", json=_DICT_FORM)
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_update_dict_without_permission_forbidden(dict_client):
    await _as_plain()
    resp = await dict_client.put("/api/v1/dict/1", json=_DICT_FORM)
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_delete_dict_without_permission_forbidden(dict_client):
    await _as_plain()
    resp = await dict_client.delete("/api/v1/dict/1")
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


# ===== 读接口登录即可访问（无 sys:dict:* 权限不拦） =====


async def test_read_endpoints_accessible_without_dict_permissions(dict_client, monkeypatch):
    await _as_plain()

    async def fake_type_page(db, page, page_size, keywords=None, status=None):
        return [], 0

    async def fake_dict_page(db, page, page_size, keywords=None, type_code=None, status=None):
        return [], 0

    async def fake_type_form(db, type_id):
        return {
            "id": type_id,
            "name": "n",
            "code": "c",
            "status": 1,
            "remark": "",
            "isPreset": False,
        }

    async def fake_dict_form(db, dict_id):
        return {
            "id": dict_id,
            "typeCode": "c",
            "name": "n",
            "value": "v",
            "status": 1,
            "defaulted": 0,
            "sort": 1,
            "remark": "",
        }

    async def fake_options(db, redis, type_code):
        return [{"value": "1", "label": "男"}]

    monkeypatch.setattr(dict_router.dict_type_service, "get_dict_type_page", fake_type_page)
    monkeypatch.setattr(dict_router.dict_type_service, "get_dict_type_form", fake_type_form)
    monkeypatch.setattr(dict_router.dict_service, "get_dict_page", fake_dict_page)
    monkeypatch.setattr(dict_router.dict_service, "get_dict_form", fake_dict_form)
    monkeypatch.setattr(dict_router.dict_service, "list_dict_options", fake_options)

    responses = [
        await dict_client.get("/api/v1/dict/types/page"),
        await dict_client.get("/api/v1/dict/types/1/form"),
        await dict_client.get("/api/v1/dict/page", params={"typeCode": "gender"}),
        await dict_client.get("/api/v1/dict/1/form"),
        await dict_client.get("/api/v1/dict/gender/options"),
    ]
    for resp in responses:
        assert resp.status_code == 200
        assert resp.json()["code"] == "00000"

    options = responses[-1].json()["data"]
    assert options == [{"value": "1", "label": "男"}]


# ===== 参数校验与路径解析 =====


async def test_create_dict_type_rejects_html_tag_in_name(dict_client):
    resp = await dict_client.post(
        "/api/v1/dict/types", json={"name": "<script>alert(1)</script>", "code": "XSS_TYPE"}
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_create_dict_type_rejects_javascript_protocol_in_code(dict_client):
    resp = await dict_client.post(
        "/api/v1/dict/types", json={"name": "合法名称", "code": "javascript:alert(1)"}
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_create_dict_rejects_missing_required_fields(dict_client):
    for form in [
        {"value": "1", "typeCode": "gender"},
        {"name": "缺value", "typeCode": "gender"},
        {"name": "缺typeCode", "value": "1"},
    ]:
        resp = await dict_client.post("/api/v1/dict", json=form)
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"


async def test_create_dict_rejects_invalid_status_value(dict_client):
    resp = await dict_client.post("/api/v1/dict", json={**_DICT_FORM, "status": 2})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_create_dict_rejects_negative_sort(dict_client):
    resp = await dict_client.post("/api/v1/dict", json={**_DICT_FORM, "sort": -1})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_page_query_rejects_invalid_pagination(dict_client):
    for params in [{"pageNum": 0}, {"pageSize": 0}, {"pageSize": 101}]:
        resp = await dict_client.get("/api/v1/dict/types/page", params=params)
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"


async def test_dict_page_requires_type_code(dict_client):
    resp = await dict_client.get("/api/v1/dict/page")
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0410"


async def test_delete_dict_types_invalid_ids_rejected(dict_client):
    resp = await dict_client.delete("/api/v1/dict/types/abc")
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"

    resp = await dict_client.delete("/api/v1/dict/abc")
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_delete_dict_types_passes_force_flag(dict_client, monkeypatch):
    captured = {}

    async def fake_delete(db, redis, type_ids, force=False):
        captured["ids"] = type_ids
        captured["force"] = force
        return True

    monkeypatch.setattr(dict_router.dict_type_service, "delete_dict_types", fake_delete)
    resp = await dict_client.delete("/api/v1/dict/types/11,22", params={"force": "true"})
    assert resp.status_code == 200
    assert resp.json()["code"] == "00000"
    assert captured["ids"] == [11, 22]
    assert captured["force"] is True
