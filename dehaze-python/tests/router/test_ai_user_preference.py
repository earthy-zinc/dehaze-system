"""用户 AI 偏好路由契约（仅需登录，无权限码）。

GET /api/v1/ai/user-preference → data.defaultModel（全局递归剔除 null ⇒ 未设置时该键不出现）；
PUT 同路径 body.defaultModel 全量替换（null/缺省=清除，空串按非法模型 A0601）。
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.dependencies.redis import get_redis, get_redis_client
from app.main import app as fastapi_app
from tests.stubs.factories import make_user_context, seed_ai_model

pytestmark = [pytest.mark.api, pytest.mark.requires_db]

_PATH = "/api/v1/ai/user-preference"
_USER_ID = 8_800_003

# conftest 的 mock_redis 会在用例运行时替换本模块的 get_redis/get_redis_client 名字，
# 依赖覆盖必须以导入期捕获的原始函数对象为键才能命中路由依赖
_ORIGINAL_GET_REDIS = get_redis
_ORIGINAL_GET_REDIS_CLIENT = get_redis_client


@pytest.fixture
async def pref_client(db, mock_redis):
    async def _override_db():
        return db

    async def _override_redis():
        return mock_redis

    async def _override_user():
        return make_user_context(_USER_ID, username="pref-user")

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[_ORIGINAL_GET_REDIS] = _override_redis
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as client:
        yield client
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(_ORIGINAL_GET_REDIS, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)


async def _override_platform(db, mock_redis) -> None:
    """保留 db/redis 覆盖、不覆盖鉴权依赖（用于未登录断言）"""

    async def _override_db():
        return db

    async def _override_redis():
        return mock_redis

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[_ORIGINAL_GET_REDIS] = _override_redis
    fastapi_app.dependency_overrides[_ORIGINAL_GET_REDIS_CLIENT] = _override_redis


async def _teardown_platform() -> None:
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(_ORIGINAL_GET_REDIS, None)
    fastapi_app.dependency_overrides.pop(_ORIGINAL_GET_REDIS_CLIENT, None)


def test_paths_registered(app):
    paths = app.openapi()["paths"]
    assert "get" in paths[_PATH]
    assert "put" in paths[_PATH]


async def test_get_unset_returns_no_default_model(pref_client):
    """未设置偏好不建行，且响应体不出现 defaultModel 键（null 被全局剔除）"""
    resp = await pref_client.get(_PATH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == "00000"
    assert "defaultModel" not in body["data"]


async def test_put_then_get_round_trips(pref_client, db, mock_redis):
    await seed_ai_model(db, mock_redis, "route-chat")
    put = await pref_client.put(_PATH, json={"defaultModel": "route-chat"})
    assert put.status_code == 200
    assert put.json()["code"] == "00000"
    get = await pref_client.get(_PATH)
    assert get.json()["data"]["defaultModel"] == "route-chat"


async def test_put_accepts_snake_case_field_name(pref_client, db, mock_redis):
    """OrmResult 同时按别名字段名与 Python 字段名解析入参"""
    await seed_ai_model(db, mock_redis, "route-chat")
    resp = await pref_client.put(_PATH, json={"default_model": "route-chat"})
    assert resp.status_code == 200
    assert resp.json()["code"] == "00000"
    assert (await pref_client.get(_PATH)).json()["data"]["defaultModel"] == "route-chat"


async def test_put_null_clears_preference(pref_client, db, mock_redis):
    await seed_ai_model(db, mock_redis, "route-chat")
    await pref_client.put(_PATH, json={"defaultModel": "route-chat"})
    resp = await pref_client.put(_PATH, json={"defaultModel": None})
    assert resp.status_code == 200
    assert "defaultModel" not in (await pref_client.get(_PATH)).json()["data"]


async def test_put_missing_field_clears_preference(pref_client, db, mock_redis):
    """缺省 defaultModel 与显式 null 等价（三端一致：全量替换语义）"""
    await seed_ai_model(db, mock_redis, "route-chat")
    await pref_client.put(_PATH, json={"defaultModel": "route-chat"})
    resp = await pref_client.put(_PATH, json={})
    assert resp.status_code == 200
    assert resp.json()["code"] == "00000"
    assert "defaultModel" not in (await pref_client.get(_PATH)).json()["data"]


async def test_put_unavailable_model_rejected(pref_client):
    resp = await pref_client.put(_PATH, json={"defaultModel": "route-ghost"})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0601"


async def test_put_empty_string_rejected(pref_client):
    """空串是"非法模型"，不是清除（与 java/go 同口径）"""
    resp = await pref_client.put(_PATH, json={"defaultModel": ""})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0601"


@pytest.mark.parametrize("method", ["get", "put"])
async def test_requires_login(db, mock_redis, method):
    """不覆盖 get_current_user：无会话一律 401（TOKEN_INVALID）"""
    await _override_platform(db, mock_redis)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=fastapi_app),
            base_url="http://test",
        ) as client:
            call = getattr(client, method)
            kwargs = {"json": {"defaultModel": None}} if method == "put" else {}
            resp = await call(_PATH, **kwargs)
    finally:
        await _teardown_platform()
    assert resp.status_code == 401
    assert resp.json()["code"] == "A0230"
