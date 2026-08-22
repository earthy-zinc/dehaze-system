import pytest

from app.infrastructure.cache.cache import CacheService
from app.service.ai.provider_health_service import (
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
