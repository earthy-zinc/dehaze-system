"""部门管理路由层测试（权限装饰器 / 参数校验 / 路径参数健壮性）。

遵循 05-python-test-rules：marker=api、dependency_overrides + monkeypatch service、
只断言业务结果。权限口径：新增/编辑/删除需 sys:dept:add|edit|delete，
列表/options/form 登录即可访问（API接口.md §2 权限标识 `-`）。
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.dependencies.redis import get_redis
from app.main import app as fastapi_app
from app.router import dept
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.api

DEPT_PERMS = ["sys:dept:add", "sys:dept:edit", "sys:dept:delete"]


def _admin_ctx():
    return make_user_context(2, username="admin", roles=["ADMIN"], permissions=DEPT_PERMS)


def _plain_ctx():
    """普通用户：无任何 sys:dept:* 权限"""
    return make_user_context(5, username="user", roles=[], permissions=[])


def _valid_form() -> dict:
    return {"name": "路由测试部门", "parentId": 1, "sort": 1, "status": 1}


@pytest.fixture
async def dept_client():
    async def _override_db():
        return object()

    async def _override_user():
        return _admin_ctx()

    async def _override_redis():
        return None

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    fastapi_app.dependency_overrides[get_redis] = _override_redis
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as client:
        yield client
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)
    fastapi_app.dependency_overrides.pop(get_redis, None)


async def _as_plain(client: AsyncClient):
    async def _override_user():
        return _plain_ctx()

    fastapi_app.dependency_overrides[get_current_user] = _override_user
    return client


# ===== 写接口越权（403 A0301，T-DPT-015/025/033） =====


async def test_create_dept_without_permission_forbidden(dept_client):
    await _as_plain(dept_client)
    resp = await dept_client.post("/api/v1/depts", json=_valid_form())
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_update_dept_without_permission_forbidden(dept_client):
    await _as_plain(dept_client)
    resp = await dept_client.put("/api/v1/depts/2", json=_valid_form())
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_delete_dept_without_permission_forbidden(dept_client):
    await _as_plain(dept_client)
    resp = await dept_client.delete("/api/v1/depts/2")
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


# ===== 读接口登录即可访问（无 sys:dept:* 权限不拦） =====


async def test_read_endpoints_accessible_without_dept_permissions(dept_client, monkeypatch):
    await _as_plain(dept_client)

    async def fake_list(db, keywords=None, status=None, current_user=None):
        return []

    async def fake_options(db, redis, current_user=None):
        return []

    async def fake_form(db, dept_id):
        return None

    monkeypatch.setattr(dept.dept_service, "get_dept_list", fake_list)
    monkeypatch.setattr(dept.dept_service, "get_dept_options", fake_options)
    monkeypatch.setattr(dept.dept_service, "get_dept_form", fake_form)

    for path in ("/api/v1/depts", "/api/v1/depts/options", "/api/v1/depts/2/form"):
        resp = await dept_client.get(path)
        assert resp.status_code == 200
        assert resp.json()["code"] == "00000"


# ===== 有权限路径放行（service 打桩，验证权限装饰器与参数解析） =====


async def test_create_dept_with_permission_ok(dept_client, monkeypatch):
    async def fake_create(db, redis, data):
        return 123

    monkeypatch.setattr(dept.dept_service, "create_dept", fake_create)
    resp = await dept_client.post("/api/v1/depts", json=_valid_form())
    assert resp.status_code == 200
    assert resp.json()["code"] == "00000"
    assert resp.json()["data"] == 123


async def test_delete_depts_parses_comma_separated_ids(dept_client, monkeypatch):
    captured = {}

    async def fake_delete(db, redis, dept_ids):
        captured["ids"] = dept_ids

    monkeypatch.setattr(dept.dept_service, "delete_depts", fake_delete)
    resp = await dept_client.delete("/api/v1/depts/2,3,4")
    assert resp.status_code == 200
    assert captured["ids"] == [2, 3, 4]


# ===== 参数校验（A0400，schema 层拦截） =====


async def test_create_dept_missing_name_rejected(dept_client):
    form = {k: v for k, v in _valid_form().items() if k != "name"}
    resp = await dept_client.post("/api/v1/depts", json=form)
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_create_dept_missing_parent_id_rejected(dept_client):
    form = {k: v for k, v in _valid_form().items() if k != "parentId"}
    resp = await dept_client.post("/api/v1/depts", json=form)
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_create_dept_invalid_status_rejected(dept_client):
    """T-DPT-007：status=99 越界"""
    resp = await dept_client.post("/api/v1/depts", json={**_valid_form(), "status": 99})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_create_dept_negative_sort_rejected(dept_client):
    """T-DPT-046：排序负数"""
    resp = await dept_client.post("/api/v1/depts", json={**_valid_form(), "sort": -1})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_create_dept_name_over_64_chars_rejected(dept_client):
    resp = await dept_client.post("/api/v1/depts", json={**_valid_form(), "name": "x" * 65})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_create_dept_xss_name_rejected(dept_client):
    resp = await dept_client.post(
        "/api/v1/depts", json={**_valid_form(), "name": "<script>alert(1)</script>"}
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_list_depts_invalid_status_filter_rejected(dept_client):
    resp = await dept_client.get("/api/v1/depts", params={"status": 99})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


# ===== 路径参数健壮性 =====


async def test_delete_depts_non_numeric_id_rejected_as_a0400(dept_client):
    """【暴露缺陷】DELETE /{ids} 内联 int(i) 解析非数字路径段抛 ValueError →
    兜底 500 B0001。契约应返回 400 A0400（参数错误）而非 500。"""
    resp = await dept_client.delete("/api/v1/depts/abc")
    assert resp.status_code == 400, (
        f"非数字 ID 应返回 400 A0400，实际 {resp.status_code}（{resp.json().get('code')}）"
    )
    assert resp.json()["code"] == "A0400"
