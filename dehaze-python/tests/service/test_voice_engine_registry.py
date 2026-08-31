"""语音引擎注册表 VoiceEngineRegistry 单元测试。

覆盖设计核心边界（易忽略）：
- 纯云端部署 default=local 但本地引擎不可用 → 抛 A0500（不降级）
- 未配置默认引擎 → 抛业务异常
- local / cloud 默认引擎 → 实例化对应 Provider
- 默认引擎解析走 Redis 短 TTL 缓存：命中且与内存实例一致不查库；
  未命中查库回填缓存；管理端改写缓存后切换即时生效
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fakeredis import FakeAsyncRedis

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.voice.provider.cloud_asr import CloudAsrProvider
from app.infrastructure.voice.provider.cloud_tts import CloudTtsProvider
from app.infrastructure.voice.provider.local_asr import LocalAsrProvider
from app.infrastructure.voice.provider.local_tts import LocalTtsProvider
from app.infrastructure.voice.provider.registry import VoiceEngineRegistry

_ENGINE_CACHE_KEY = "voice:engine:{}"


class _FakeSession:
    """假 async 上下文管理器：registry 经 session_factory 进入时返回假 db"""

    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


def _row(provider_id: int, code: str, engine_type: str, *, status: int = 1, is_default: int = 1):
    return SimpleNamespace(
        id=provider_id, provider_code=code, engine_type=engine_type,
        status=status, is_default=is_default,
    )


def _make_registry(
    default_row=None,
    engine_available=True,
    *,
    by_id: dict[int, SimpleNamespace] | None = None,
    redis: FakeAsyncRedis | None = None,
) -> tuple[VoiceEngineRegistry, FakeAsyncRedis, SimpleNamespace]:
    """构造注入桩依赖的 registry，返回 (registry, fakeredis, repo桩) 便于断言"""
    redis = redis or FakeAsyncRedis(decode_responses=True)
    repo = SimpleNamespace(
        get_default=AsyncMock(return_value=default_row),
        get_by_id=AsyncMock(side_effect=lambda db, pid: (by_id or {}).get(pid)),
    )

    async def _redis_factory():
        return redis

    registry = VoiceEngineRegistry(
        repository=repo,
        session_factory=lambda: _FakeSession(),
        engine_available=lambda engine_type: engine_available,
        redis_factory=_redis_factory,
    )
    return registry, redis, repo


@pytest.mark.asyncio
async def test_default_local_but_engine_unavailable_raises_a0500():
    """纯云端部署：default=local 且本地引擎不可用 → 抛 A0500（不降级）"""
    registry, _, _ = _make_registry(_row(1, "local", "asr"), engine_available=False)

    with pytest.raises(BusinessException) as exc:
        await registry.get_asr_provider()

    assert exc.value.code == ResultCode.BUSINESS_ERROR  # A0500


@pytest.mark.asyncio
async def test_no_default_engine_raises():
    """未配置默认引擎 → 抛业务异常（提示检查 sys_voice_provider）"""
    registry, _, _ = _make_registry(None)

    with pytest.raises(BusinessException) as exc:
        await registry.get_tts_provider()

    assert exc.value.code == ResultCode.BUSINESS_ERROR


@pytest.mark.asyncio
async def test_local_default_returns_local_provider():
    """default=local 且本地可用 → 实例化 LocalAsrProvider / LocalTtsProvider"""
    r1, _, _ = _make_registry(_row(1, "local", "asr"))
    assert isinstance(await r1.get_asr_provider(), LocalAsrProvider)
    r2, _, _ = _make_registry(_row(2, "local", "tts"))
    assert isinstance(await r2.get_tts_provider(), LocalTtsProvider)


@pytest.mark.asyncio
async def test_cloud_default_returns_cloud_provider():
    """default=云端 → 实例化 CloudAsrProvider / CloudTtsProvider"""
    r1, _, _ = _make_registry(_row(1, "aliyun", "asr"))
    assert isinstance(await r1.get_asr_provider(), CloudAsrProvider)
    r2, _, _ = _make_registry(_row(2, "azure", "tts"))
    assert isinstance(await r2.get_tts_provider(), CloudTtsProvider)


@pytest.mark.asyncio
async def test_resolve_backfills_redis_cache():
    """未命中缓存查库解析后回填 Redis（voice:engine:{engine_type}，JSON 含 provider_id/provider_code）"""
    registry, redis, repo = _make_registry(_row(7, "local", "asr"))

    await registry.get_asr_provider()

    raw = await redis.get(_ENGINE_CACHE_KEY.format("asr"))
    assert json.loads(raw) == {"provider_id": 7, "provider_code": "local"}
    repo.get_default.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_hit_reuses_provider_without_db():
    """缓存命中且与内存实例一致：不查库直接复用 Provider"""
    registry, redis, repo = _make_registry(_row(7, "local", "asr"))

    await registry.get_asr_provider()
    await registry.get_asr_provider()

    # 仅首次解析查库一次（回源 get_default），缓存命中路径不再触达 repository
    repo.get_default.assert_awaited_once()
    repo.get_by_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_default_switch_takes_effect_immediately():
    """管理端切换默认引擎（改写 Redis 缓存）→ 下次解析即时路由到新引擎，不重启"""
    local_row = _row(1, "local", "asr")
    cloud_row = _row(2, "aliyun", "asr")
    registry, redis, repo = _make_registry(
        local_row, by_id={1: local_row, 2: cloud_row}
    )
    assert isinstance(await registry.get_asr_provider(), LocalAsrProvider)

    # 模拟管理端切换：失效 + 新默认引擎信息写入缓存（VoiceAdminService 失效后
    # 首次解析回源 get_default 也会回填，此处直接预置新引擎缓存覆盖同一语义）
    await redis.set(
        _ENGINE_CACHE_KEY.format("asr"),
        json.dumps({"provider_id": 2, "provider_code": "aliyun"}),
    )

    provider = await registry.get_asr_provider()
    assert isinstance(provider, CloudAsrProvider)
    # 缓存命中按 provider_id 精确取行，不回源 get_default
    repo.get_default.assert_awaited_once()


@pytest.mark.asyncio
async def test_stale_cache_entry_falls_back_to_default_query():
    """缓存指向的引擎已禁用 → 视为未命中，回源 get_default 重写缓存"""
    registry, redis, repo = _make_registry(_row(7, "aliyun", "tts"))
    await redis.set(
        _ENGINE_CACHE_KEY.format("tts"),
        json.dumps({"provider_id": 7, "provider_code": "aliyun"}),
    )
    # get_by_id 返回禁用行（status=0）
    repo.get_by_id = AsyncMock(return_value=_row(7, "aliyun", "tts", status=0))

    assert isinstance(await registry.get_tts_provider(), CloudTtsProvider)
    repo.get_default.assert_awaited_once()
    raw = await redis.get(_ENGINE_CACHE_KEY.format("tts"))
    assert json.loads(raw) == {"provider_id": 7, "provider_code": "aliyun"}


@pytest.mark.asyncio
async def test_invalidate_default_cache_deletes_key():
    """管理端 is_default/status 变更后失效缓存：删除对应 engine_type 的 Redis key"""
    registry, redis, _ = _make_registry(_row(7, "local", "asr"))
    await redis.set(_ENGINE_CACHE_KEY.format("asr"), json.dumps({"provider_id": 7}))

    await registry.invalidate_default_cache("asr")

    assert await redis.get(_ENGINE_CACHE_KEY.format("asr")) is None


@pytest.mark.asyncio
async def test_invalidate_swallows_redis_failure():
    """Redis 异常时失效操作不抛错（管理端操作不因缓存失效失败而阻断）"""

    async def _broken_factory():
        raise ConnectionError("redis down")

    registry = VoiceEngineRegistry(
        repository=SimpleNamespace(),
        session_factory=lambda: _FakeSession(),
        engine_available=lambda t: True,
        redis_factory=_broken_factory,
    )

    await registry.invalidate_default_cache("asr")  # 不抛异常即通过
