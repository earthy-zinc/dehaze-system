"""运营统计服务聚合测试（真实 db fixture + mock_redis）"""

import pytest

from app.models.entity.sys_ai_provider import SysAiProvider
from app.models.schema.ai_provider import UsageStatsQuery
from app.repository.ai_billing_repository import ai_billing_repository
from app.repository.ai_provider_repository import ai_provider_repository
from app.service.ai_usage_stats_service import ai_usage_stats_service

pytestmark = pytest.mark.requires_db


async def _seed_provider(db, **overrides) -> SysAiProvider:
    data = dict(
        provider_code="openai",
        display_name="OpenAI",
        api_base_url="https://api.openai.com/v1",
        protocol_type="openai_compat",
        auth_type="bearer",
        sort_order=0,
        health_check_enabled=1,
        status=1,
    )
    data.update(overrides)
    return await ai_provider_repository.create(db, SysAiProvider(**data))


async def _seed_billing(db, **overrides):
    data = dict(
        user_id=1,
        model="gpt-4o",
        bill_type="chat",
        input_tokens=100,
        output_tokens=50,
        credits=30,
        latency_ms=500,
        actual_model=None,
    )
    data.update(overrides)
    return await ai_billing_repository.create_billing(db, **data)


async def test_provider_health_aggregation(db, mock_redis):
    await _seed_provider(db, provider_code="prov_health", display_name="健康供应商")
    result = await ai_usage_stats_service.get_usage_stats(
        db, mock_redis, UsageStatsQuery()
    )
    prov = next(p for p in result.provider_health if p.provider_name == "健康供应商")
    assert prov.provider_id > 0
    assert prov.health in ("healthy", "suspicious", "open")
    assert prov.call_count >= 0
    assert 0 <= prov.success_rate <= 1
    assert isinstance(prov.circuit_open, bool)


async def test_model_usage_aggregates_billing(db, mock_redis):
    await _seed_billing(db, model="gpt-4o", input_tokens=100, output_tokens=50, credits=30)
    await _seed_billing(db, model="gpt-4o", input_tokens=200, output_tokens=100, credits=60)
    result = await ai_usage_stats_service.get_usage_stats(db, mock_redis, UsageStatsQuery())
    gpt = next(m for m in result.model_usage if m.model_id == "gpt-4o")
    assert gpt.call_count == 2
    assert gpt.input_tokens == 300
    assert gpt.output_tokens == 150
    assert gpt.credits == 90


async def test_downgrade_frequency_by_original_model(db, mock_redis):
    await _seed_billing(db, model="gpt-4o-mini", actual_model="gpt-4o")
    await _seed_billing(db, model="gpt-4o-mini", actual_model="gpt-4o")
    await _seed_billing(db, model="gpt-4o", actual_model=None)
    result = await ai_usage_stats_service.get_usage_stats(db, mock_redis, UsageStatsQuery())
    downgrade = next(d for d in result.degrade_fault.downgrade_frequency if d.model_id == "gpt-4o")
    assert downgrade.count == 2


async def test_key_failover_count_from_redis(db, mock_redis):
    await _seed_provider(db, provider_code="prov_failover")
    # 两个 Key 处于冷却期（失败切换后的临时不可用状态）
    await mock_redis.set("ai:provider_key:101:unavailable", 1)
    await mock_redis.set("ai:provider_key:102:unavailable", 1)
    result = await ai_usage_stats_service.get_usage_stats(db, mock_redis, UsageStatsQuery())
    assert result.degrade_fault.key_failover_count == 2
