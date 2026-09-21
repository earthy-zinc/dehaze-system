"""图像输入历史记录路由层测试（参数校验 / 批量删除表单 / 越权路径）。

遵循 05-python-test-rules：marker=api、dependency_overrides + monkeypatch service、
只断言业务结果。权限口径：历史记录为用户个人数据，全部接口仅要求登录态，
数据隔离靠 user_id 过滤（无 image:* 权限码，见模块 API 文档 §3）。
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.router import image_input
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.api


def _admin_ctx():
    return make_user_context(1, username="admin", roles=["ADMIN"], permissions=["*"])


def _plain_ctx():
    return make_user_context(5, username="user", roles=[], permissions=[])


def _valid_form() -> dict:
    return {
        "originalImageUrl": "/images/test_haze.jpg",
        "algorithmId": 1,
        "algorithmName": "DCP",
        "processingTime": 1520,
        "status": 1,
        "inputSource": "upload",
    }


@pytest.fixture
async def history_client():
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


# ===== 参数校验（对抗性输入，A0400）=====


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", 99),
        ("status", 0),
        ("inputSource", "hacked"),
        ("processingTime", -100),
    ],
)
async def test_create_rejects_invalid_enum_and_range(history_client, field, value):
    form = _valid_form()
    form[field] = value
    resp = await history_client.post("/api/v1/image-input/history", json=form)
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


@pytest.mark.parametrize(
    ("field", "length"),
    [("originalImageUrl", 501), ("algorithmName", 101)],
)
async def test_create_rejects_overlength_fields(history_client, field, length):
    form = _valid_form()
    form[field] = "a" * length
    resp = await history_client.post("/api/v1/image-input/history", json=form)
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_create_rejects_invalid_algorithm_params_json(history_client):
    """algorithm_params 落库到 MySQL JSON 列，非法 JSON 会在 Java 端 SQL 报错，后端前置拦截"""
    form = _valid_form()
    form["algorithmParams"] = "not-a-json{["
    resp = await history_client.post("/api/v1/image-input/history", json=form)
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_batch_delete_rejects_empty_ids(history_client):
    resp = await history_client.request(
        "DELETE", "/api/v1/image-input/history/batch", json={"ids": []}
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_batch_delete_rejects_non_integer_ids(history_client):
    resp = await history_client.request(
        "DELETE", "/api/v1/image-input/history/batch", json={"ids": ["abc"]}
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_batch_delete_uses_typed_form_not_raw_dict(history_client, monkeypatch):
    """批量删除必须走 BatchDeleteForm 强类型校验（历史实现用裸 dict 绕过了校验）"""
    captured = {}

    async def _fake_batch_delete(db, ids, user_id):
        captured["ids"] = ids
        captured["user_id"] = user_id
        return len(ids)

    monkeypatch.setattr(image_input.input_history_service, "batch_delete", _fake_batch_delete)
    resp = await history_client.request(
        "DELETE", "/api/v1/image-input/history/batch", json={"ids": [1, 2, 3]}
    )
    assert resp.status_code == 200
    assert resp.json()["data"] == 3
    assert captured["ids"] == [1, 2, 3]
    assert captured["user_id"] == 1


# ===== 业务异常 =====


async def test_get_history_not_found(history_client, monkeypatch):
    async def _none(db, history_id, user_id):
        return None

    monkeypatch.setattr(image_input.input_history_service, "get_history", _none)
    resp = await history_client.get("/api/v1/image-input/history/99999")
    assert resp.status_code == 400
    assert resp.json()["code"] == ResultCode.RESOURCE_NOT_FOUND.code


async def test_get_history_rejects_non_integer_id(history_client):
    resp = await history_client.get("/api/v1/image-input/history/not-an-int")
    assert resp.status_code == 400


async def test_delete_history_undefined_error_raises_404(history_client, monkeypatch):
    """路由在服务返回 None 时抛 A0401（不泄露他人记录存在性由服务层保证）"""

    async def _raise(db, history_id, user_id):
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "历史记录不存在")

    monkeypatch.setattr(image_input.input_history_service, "get_history", _raise)
    resp = await history_client.get("/api/v1/image-input/history/1")
    assert resp.json()["code"] == ResultCode.RESOURCE_NOT_FOUND.code


# ===== 静态/动态路由顺序（/batch、/clear 不得被 /{id} 吞掉）=====


async def test_static_batch_route_not_swallowed_by_path_param(history_client, monkeypatch):
    async def _fake_batch_delete(db, ids, user_id):
        return 0

    monkeypatch.setattr(image_input.input_history_service, "batch_delete", _fake_batch_delete)
    resp = await history_client.request(
        "DELETE", "/api/v1/image-input/history/batch", json={"ids": [1]}
    )
    assert resp.status_code == 200


async def test_static_clear_route_not_swallowed_by_path_param(history_client, monkeypatch):
    async def _fake_clear(db, user_id):
        return 0

    monkeypatch.setattr(image_input.input_history_service, "clear_history", _fake_clear)
    resp = await history_client.delete("/api/v1/image-input/history/clear")
    assert resp.status_code == 200
