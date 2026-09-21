"""CacheService 多级缓存测试：L2 回填 L1 的 TTL 一致性。

L1 回填若不继承 L2 剩余 TTL（退化为 L1 默认 TTL），L2 过期后 L1 仍返回旧值，
造成"改了数据必须重启后端"类脏读，此处锁定回填语义。
"""

import json
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
    assert svc._l1 is not None
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
    assert svc._l1 is not None
    assert svc._l1.get("k5") is None


# ── 缓存失效广播：L1 失效 + MCP 变更后的推理图缓存失效 ─────────────────────────


async def test_graph_invalidation_message_clears_reasoning_graph_cache(monkeypatch):
    """java/go 原生改 MCP 后广播：收到即失效本实例图缓存（已构图仍持旧工具集）"""
    from app.infrastructure.cache import cache as cache_mod
    from app.service.ai.service import reasoning_service as rs_mod

    service = rs_mod.ReasoningService()
    service._graphs[(1, 2, "m")] = object()
    monkeypatch.setattr(rs_mod, "reasoning_service", service)

    await cache_mod._handle_invalidation_message(
        json.dumps({"type": "ai_graph_invalidate", "senderId": "dehaze-java-abc"})
    )

    assert service._graphs == {}


async def test_graph_invalidation_ignores_own_message(monkeypatch):
    """忽略自己发送的消息（防自消费）：本实例已在变更处直接失效过"""
    from app.infrastructure.cache import cache as cache_mod
    from app.service.ai.service import reasoning_service as rs_mod

    service = rs_mod.ReasoningService()
    service._graphs[(1, 2, "m")] = object()
    monkeypatch.setattr(rs_mod, "reasoning_service", service)

    await cache_mod._handle_invalidation_message(
        json.dumps({"type": "ai_graph_invalidate", "senderId": cache_mod._INSTANCE_ID})
    )

    assert list(service._graphs) == [(1, 2, "m")]


async def test_key_message_still_clears_l1():
    """图缓存消息分支不得挤掉原有 L1 失效语义"""
    from app.infrastructure.cache import cache as cache_mod

    l1 = cache_mod._get_shared_l1()
    assert l1 is not None, "L1 未启用则本用例无意义"
    l1.set("ai:agent:default", "v")

    await cache_mod._handle_invalidation_message(
        json.dumps({"type": "key", "key": "ai:agent:default", "senderId": "dehaze-go"})
    )

    assert l1.get("ai:agent:default") is None


async def test_publish_graph_invalidation_payload(mock_redis, monkeypatch):
    """发布载荷为 {type, senderId}（无 key）：载荷结构即跨端协议，java/go 据此对齐"""
    from app.infrastructure.cache import cache as cache_mod

    published: list[tuple[str, str]] = []

    async def _spy(channel: str, message: str) -> int:
        published.append((channel, message))
        return 1

    monkeypatch.setattr(mock_redis, "publish", _spy)

    await cache_mod.publish_graph_invalidation()

    assert len(published) == 1
    channel, payload = published[0]
    assert channel == settings.CACHE_INVALIDATION_CHANNEL
    msg = json.loads(payload)
    assert msg["type"] == "ai_graph_invalidate"
    assert "key" not in msg
    assert msg["senderId"] == cache_mod._INSTANCE_ID
