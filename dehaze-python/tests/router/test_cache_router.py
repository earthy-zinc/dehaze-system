"""缓存统一失效入口路由测试（POST /api/v1/cache/clear）。

权限口径：仅 ROOT/ADMIN（is_admin）可调用，普通用户 403 A0301；
清理必须经 CacheService（同步清 L1+L2 并广播），裸删 Redis 不会清 L1——
通过"经 CacheService 写入后 L1 有值，调接口后 L1 被清"来锁定该约束。
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.dependencies.auth import get_current_user
from app.dependencies.redis import get_redis
from app.infrastructure.cache.cache import CacheService
from app.main import app as fastapi_app
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.api

# conftest mock_redis 的动态扫描会把本模块顶层 get_redis 名字替换为桩函数，
# 依赖覆盖必须以导入期捕获的原始函数对象为键才能命中路由依赖
_ORIGINAL_GET_REDIS = get_redis


def _root_ctx():
    return make_user_context(1, username="admin", roles=["ROOT"], permissions=[])


def _admin_ctx():
    return make_user_context(2, username="admin2", roles=["ADMIN"], permissions=[])


def _user_ctx():
    return make_user_context(5, username="user", roles=[], permissions=[])


@pytest.fixture
async def cache_client(mock_redis):
    svc = CacheService(mock_redis)  # 与路由共用进程级 L1 单例
    if svc._l1 is not None:
        svc._l1.clear()

    async def _override_user():
        return _root_ctx()

    async def _override_redis():
        return mock_redis

    fastapi_app.dependency_overrides[get_current_user] = _override_user
    fastapi_app.dependency_overrides[_ORIGINAL_GET_REDIS] = _override_redis
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app), base_url="http://test"
    ) as client:
        yield client, svc, mock_redis
    fastapi_app.dependency_overrides.pop(get_current_user, None)
    fastapi_app.dependency_overrides.pop(_ORIGINAL_GET_REDIS, None)


async def test_clear_single_key(cache_client):
    client, svc, redis = cache_client
    await svc.set("menu:routes", '{"routes": []}', ttl=600)

    resp = await client.post("/api/v1/cache/clear", json={"key": "menu:routes"})
    assert resp.status_code == 200
    assert resp.json()["code"] == "00000"
    assert resp.json()["data"] == [{"target": "menu:routes", "deleted": 1}]
    assert await redis.get("menu:routes") is None
    assert svc._l1.get("menu:routes") is None


async def test_clear_pattern(cache_client):
    client, svc, redis = cache_client
    await svc.set("role:perms:ADMIN", '["*"]', ttl=600)
    await svc.set("role:perms:USER", '["sys:user:list"]', ttl=600)
    await svc.set("other:key", "keep", ttl=600)

    resp = await client.post("/api/v1/cache/clear", json={"pattern": "role:perms:*"})
    assert resp.status_code == 200
    assert resp.json()["data"] == [{"target": "role:perms:*", "deleted": 2}]
    assert await redis.get("role:perms:ADMIN") is None
    assert await redis.get("other:key") == "keep"
    assert svc._l1.get("role:perms:ADMIN") is None


async def test_clear_all_default_clears_business_caches(cache_client):
    client, svc, redis = cache_client
    await svc.set("menu:routes", "[]", ttl=600)
    await svc.set("role:perms:ADMIN", "[]", ttl=600)
    await svc.set("dict:options:gender", "[]", ttl=600)
    await svc.set("session:abc", "keep-session", ttl=600)

    resp = await client.post("/api/v1/cache/clear", json={})
    assert resp.status_code == 200
    targets = {item["target"]: item["deleted"] for item in resp.json()["data"]}
    assert targets["menu:routes"] == 1
    assert targets["role:perms:*"] == 1
    assert targets["dict:options:*"] == 1
    # session 等基础设施 key 不在业务缓存清单内，不受影响
    assert await redis.get("session:abc") == "keep-session"
    assert svc._l1.get("menu:routes") is None


async def test_clear_admin_role_allowed(cache_client):
    client, _, _ = cache_client

    async def _override_user():
        return _admin_ctx()

    fastapi_app.dependency_overrides[get_current_user] = _override_user
    resp = await client.post("/api/v1/cache/clear", json={})
    assert resp.status_code == 200


async def test_clear_ordinary_user_forbidden(cache_client):
    client, _, _ = cache_client

    async def _override_user():
        return _user_ctx()

    fastapi_app.dependency_overrides[get_current_user] = _override_user
    resp = await client.post("/api/v1/cache/clear", json={})
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_clear_key_and_pattern_conflict(cache_client):
    client, _, _ = cache_client
    resp = await client.post("/api/v1/cache/clear", json={"key": "a", "pattern": "b"})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_clear_reject_wildcard_pattern(cache_client):
    """pattern=* 会清掉 session/限流等基础设施 key，必须拒绝；全清走空 body 枚举路径"""
    client, svc, redis = cache_client
    await svc.set("session:abc", "keep", ttl=600)
    resp = await client.post("/api/v1/cache/clear", json={"pattern": "*"})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"
    assert await redis.get("session:abc") == "keep"
