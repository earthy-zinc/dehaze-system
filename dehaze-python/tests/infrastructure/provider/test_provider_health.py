import pytest

from app.infrastructure.cache.cache import CacheService
from app.infrastructure.provider import provider_health_service as phs
from app.infrastructure.provider.provider_health_service import (
    _CIRCUIT_KEY,
    _PROBE_KEY,
    _RECOVERY_KEY,
    _THRESHOLDS_KEY,
    provider_health_service,
    set_health_check_enabled,
)

_SEED = {
    "error_rate_warn": 0.1,
    "error_rate_open": 0.3,
    "min_window_calls": 20,
    "consecutive_failures": 5,
    "circuit_cooldown": 60,
}


@pytest.fixture(autouse=True)
def _reset_thresholds_cache():
    """阈值进程内缓存跨用例残留会让上例的阈值泄漏给下例"""
    phs._thresholds_cache = None
    yield
    phs._thresholds_cache = None


async def _seed_thresholds(redis) -> None:
    await CacheService(redis).set_json(_THRESHOLDS_KEY, _SEED, 300)


async def test_consecutive_failures_opens_circuit(mock_redis):
    await _seed_thresholds(mock_redis)
    provider_id = 1
    for _ in range(4):
        await provider_health_service.record_call(mock_redis, provider_id, False, "500", 120)
    assert await mock_redis.get("ai:provider:1:circuit_open") is None

    await provider_health_service.record_call(mock_redis, provider_id, False, "500", 120)
    assert await mock_redis.get("ai:provider:1:circuit_open") is not None
    assert await mock_redis.ttl("ai:provider:1:circuit_open") == 60
    assert await provider_health_service.get_status(mock_redis, provider_id) == "open"


async def test_error_rate_opens_circuit(mock_redis):
    await _seed_thresholds(mock_redis)
    provider_id = 2
    for i in range(25):
        success = i % 3 != 0
        await provider_health_service.record_call(
            mock_redis, provider_id, success, None if success else "500", 100
        )
    assert await mock_redis.get("ai:provider:2:circuit_open") is not None


async def test_state_machine_healthy(mock_redis):
    await _seed_thresholds(mock_redis)
    for _ in range(20):
        await provider_health_service.record_call(mock_redis, 3, True, None, 80)
    snapshot = await provider_health_service.get_health_snapshot(mock_redis, 3)
    assert snapshot["status"] == "healthy"
    assert await provider_health_service.get_status(mock_redis, 3) == "healthy"


async def test_state_machine_suspicious(mock_redis):
    await _seed_thresholds(mock_redis)
    provider_id = 4
    for _ in range(20):
        await provider_health_service.record_call(mock_redis, provider_id, True, None, 80)
    for _ in range(4):
        await provider_health_service.record_call(mock_redis, provider_id, False, "500", 120)
    snapshot = await provider_health_service.get_health_snapshot(mock_redis, provider_id)
    assert snapshot["status"] == "suspicious"
    assert snapshot["error_rate"] == pytest.approx(4 / 24, abs=1e-4)
    assert await provider_health_service.get_status(mock_redis, provider_id) == "suspicious"


async def test_circuit_open_ttl_set(mock_redis):
    await _seed_thresholds(mock_redis)
    for _ in range(5):
        await provider_health_service.record_call(mock_redis, 5, False, "429", 90)
    assert await mock_redis.ttl("ai:provider:5:circuit_open") == 60


async def test_health_check_disabled_never_opens(mock_redis):
    await _seed_thresholds(mock_redis)
    provider_id = 6
    await set_health_check_enabled(mock_redis, provider_id, False)
    for _ in range(10):
        await provider_health_service.record_call(mock_redis, provider_id, False, "500", 100)
    assert await mock_redis.get("ai:provider:6:circuit_open") is None
    assert await provider_health_service.get_status(mock_redis, provider_id) == "healthy"


async def test_close_circuit_manual(mock_redis):
    await _seed_thresholds(mock_redis)
    provider_id = 7
    for _ in range(5):
        await provider_health_service.record_call(mock_redis, provider_id, False, "500", 100)
    assert await mock_redis.get("ai:provider:7:circuit_open") is not None

    await provider_health_service.close_circuit(mock_redis, provider_id)
    assert await mock_redis.get("ai:provider:7:circuit_open") is None
    assert await mock_redis.get("ai:provider:7:fail_streak") is None
    assert await provider_health_service.get_status(mock_redis, provider_id) != "open"


async def test_success_resets_fail_streak(mock_redis):
    await _seed_thresholds(mock_redis)
    provider_id = 8
    for _ in range(3):
        await provider_health_service.record_call(mock_redis, provider_id, False, "500", 100)
    assert await mock_redis.get("ai:provider:8:fail_streak") is not None
    await provider_health_service.record_call(mock_redis, provider_id, True, None, 100)
    assert await mock_redis.get("ai:provider:8:fail_streak") is None


# ── 半开渐进恢复（冷却结束不得全量放行）────────────────


async def _open_circuit(mock_redis, provider_id: int) -> None:
    for _ in range(5):
        await provider_health_service.record_call(mock_redis, provider_id, False, "500", 100)
    assert await mock_redis.get(_CIRCUIT_KEY.format(provider_id)) is not None


def _expire_cooldown(mock_redis, provider_id: int):
    """模拟冷却期结束（Redis TTL 到期自动清除熔断标记）"""
    return mock_redis.delete(_CIRCUIT_KEY.format(provider_id))


async def test_recovery_marker_outlives_cooldown(mock_redis):
    await _seed_thresholds(mock_redis)
    await _open_circuit(mock_redis, 11)
    assert await mock_redis.ttl(_CIRCUIT_KEY.format(11)) == 60
    assert await mock_redis.ttl(_RECOVERY_KEY.format(11)) == 120


async def test_half_open_releases_only_one_probe(mock_redis):
    await _seed_thresholds(mock_redis)
    await _open_circuit(mock_redis, 12)
    await _expire_cooldown(mock_redis, 12)

    assert await provider_health_service.get_status(mock_redis, 12) == "half_open"
    assert await mock_redis.ttl(_PROBE_KEY.format(12)) == 60
    # 探测在飞：后续请求继续阻断，不得全量放行
    assert await provider_health_service.get_status(mock_redis, 12) == "open"


async def test_half_open_probe_success_fully_recovers(mock_redis):
    await _seed_thresholds(mock_redis)
    await _open_circuit(mock_redis, 13)
    await _expire_cooldown(mock_redis, 13)
    assert await provider_health_service.get_status(mock_redis, 13) == "half_open"

    await provider_health_service.record_call(mock_redis, 13, True, None, 80)
    assert await mock_redis.get(_CIRCUIT_KEY.format(13)) is None
    assert await mock_redis.get(_RECOVERY_KEY.format(13)) is None
    assert await mock_redis.get(_PROBE_KEY.format(13)) is None
    assert await provider_health_service.get_status(mock_redis, 13) == "healthy"


async def test_half_open_probe_failure_reopens(mock_redis):
    await _seed_thresholds(mock_redis)
    await _open_circuit(mock_redis, 14)
    await _expire_cooldown(mock_redis, 14)
    assert await provider_health_service.get_status(mock_redis, 14) == "half_open"

    await provider_health_service.record_call(mock_redis, 14, False, "500", 100)
    assert await mock_redis.ttl(_CIRCUIT_KEY.format(14)) == 60
    assert await provider_health_service.get_status(mock_redis, 14) == "open"


async def test_healthy_provider_never_takes_probe_lease(mock_redis):
    """未熔断供应商不得因探测租约被误判为阻断"""
    await _seed_thresholds(mock_redis)
    await provider_health_service.record_call(mock_redis, 15, True, None, 80)
    assert await mock_redis.get(_PROBE_KEY.format(15)) is None
    assert await provider_health_service.get_status(mock_redis, 15) == "healthy"


async def test_close_circuit_drops_recovery_and_probe(mock_redis):
    await _seed_thresholds(mock_redis)
    await _open_circuit(mock_redis, 16)
    await _expire_cooldown(mock_redis, 16)
    await provider_health_service.get_status(mock_redis, 16)

    await provider_health_service.close_circuit(mock_redis, 16)
    assert await mock_redis.get(_RECOVERY_KEY.format(16)) is None
    assert await mock_redis.get(_PROBE_KEY.format(16)) is None
    assert await provider_health_service.get_status(mock_redis, 16) == "healthy"


# ── 调用窗口分桶（判定开销与调用量无关）────────────────


async def test_window_counts_ignores_buckets_beyond_24h(mock_redis):
    await _seed_thresholds(mock_redis)
    provider_id = 17
    now_ms = phs._now_ms()
    bucket = now_ms // 1000 // phs._BUCKET_SECONDS
    key = phs._WINDOW_KEY.format(provider_id)
    stale = bucket - phs._WINDOW_BUCKETS
    await mock_redis.hset(
        key,
        mapping={
            f"{bucket}:t": 10,
            f"{bucket}:f": 3,
            f"{bucket}:l": 1,
            f"{stale}:t": 99,
            f"{stale}:f": 99,
            f"{stale}:l": 99,
        },
    )
    assert await provider_health_service._window_counts(
        mock_redis, provider_id, now_ms, return_detail=True
    ) == (10, 3, 1)


async def test_snapshot_reports_window_totals_without_cap(mock_redis):
    """窗口计数为累加值，不因保留条数上限而少算（旧实现上限 5000 条）"""
    await _seed_thresholds(mock_redis)
    provider_id = 18
    for _ in range(3):
        await provider_health_service.record_call(mock_redis, provider_id, True, None, 80)
    for _ in range(2):
        await provider_health_service.record_call(mock_redis, provider_id, False, "500", 90)
    snapshot = await provider_health_service.get_health_snapshot(mock_redis, provider_id)
    assert snapshot["total_calls_24h"] == 5
    assert snapshot["error_rate"] == pytest.approx(0.4, abs=1e-4)


async def test_limit_outcome_counted_as_failure_and_limit(mock_redis):
    await _seed_thresholds(mock_redis)
    provider_id = 19
    await provider_health_service.record_call(mock_redis, provider_id, False, "429", 70)
    await provider_health_service.record_call(mock_redis, provider_id, True, None, 70)
    snapshot = await provider_health_service.get_health_snapshot(mock_redis, provider_id)
    assert snapshot["error_rate"] == pytest.approx(0.5, abs=1e-4)
    assert snapshot["limit_rate"] == pytest.approx(0.5, abs=1e-4)


# ── 阈值进程内缓存（失败路径每次判定都要读阈值）────────


async def test_thresholds_process_cache_skips_redis_reread(mock_redis):
    await _seed_thresholds(mock_redis)
    first = await phs._load_thresholds(mock_redis)
    await CacheService(mock_redis).delete(_THRESHOLDS_KEY)

    cached = await phs._load_thresholds(mock_redis)
    assert cached.consecutive_failures == first.consecutive_failures
    assert await CacheService(mock_redis).get_json(_THRESHOLDS_KEY) is None  # 未回源


async def test_thresholds_reloaded_after_process_cache_cleared(mock_redis):
    await _seed_thresholds(mock_redis)
    await phs._load_thresholds(mock_redis)
    await CacheService(mock_redis).set_json(
        _THRESHOLDS_KEY, {**_SEED, "consecutive_failures": 9}, 300
    )
    phs._thresholds_cache = None

    assert (await phs._load_thresholds(mock_redis)).consecutive_failures == 9
