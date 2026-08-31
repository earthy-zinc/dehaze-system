"""CacheService 多级缓存测试：L2 回填 L1 的 TTL 一致性。

L1 回填若不继承 L2 剩余 TTL（退化为 L1 默认 TTL），L2 过期后 L1 仍返回旧值，
造成"改了数据必须重启后端"类脏读，此处锁定回填语义。
"""

import time

import pytest

from app.config import settings
from app.infrastructure.cache.cache import CacheService
from app.infrastructure.cache.local_cache import TTLCache

pytestmark = pytest.mark.unit

_L1_TTL = settings.CACHE_L1_TTL


def _fresh_service(mock_redis) -> CacheService:
    """独立 L1 实例（进程级共享单例会跨测试残留）"""
    svc = CacheService(mock_redis)
    svc._l1 = TTLCache(maxsize=100, default_ttl=300)
    return svc


def _l1_remaining_ttl(svc: CacheService, key: str) -> float:
    _, expire_at = svc._l1._cache[key]
    return expire_at - time.monotonic()


async def test_get_backfills_l1_with_l2_remaining_ttl(mock_redis):
    svc = _fresh_service(mock_redis)
    await mock_redis.setex("k1", 600, "v")
    assert await svc.get("k1") == "v"
    assert 590 < _l1_remaining_ttl(svc, "k1") <= 600


async def test_get_backfills_l1_falls_back_when_l2_no_expire(mock_redis):
    svc = _fresh_service(mock_redis)
    await mock_redis.set("k2", "v")  # 无过期时间，TTL 返回 -1
    assert await svc.get("k2") == "v"
    assert _L1_TTL - 5 < _l1_remaining_ttl(svc, "k2") <= _L1_TTL


async def test_get_with_loader_backfills_l1_with_l2_remaining_ttl(mock_redis):
    svc = _fresh_service(mock_redis)
    await mock_redis.setex("k3", 600, "v")

    async def loader():
        raise AssertionError("L2 命中时不应触发 loader")

    assert await svc.get_with_loader("k3", loader) == "v"
    assert 590 < _l1_remaining_ttl(svc, "k3") <= 600


async def test_get_json_with_loader_backfills_l1_with_l2_remaining_ttl(mock_redis):
    svc = _fresh_service(mock_redis)
    await mock_redis.setex("k4", 600, '{"a": 1}')

    async def loader():
        raise AssertionError("L2 命中时不应触发 loader")

    assert await svc.get_json_with_loader("k4", loader) == {"a": 1}
    assert 590 < _l1_remaining_ttl(svc, "k4") <= 600


async def test_delete_clears_l1_and_l2(mock_redis):
    """清理必须经 CacheService（同步删 L1+L2 并广播），裸删 Redis 不会清 L1"""
    svc = _fresh_service(mock_redis)
    await svc.set("k5", "v", ttl=600)
    assert await svc.get("k5") == "v"

    assert await svc.delete("k5") is True
    assert await mock_redis.get("k5") is None
    assert svc._l1.get("k5") is None
