from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.requires_db

from app.core.exceptions import BusinessException
from app.service import api_key_service as aks
from app.service.ai.service.compatible_governance import (
    compatible_governance_service,
    GovernanceError,
)
from app.service.api_key_service import api_key_service


def _api_key(**overrides) -> SimpleNamespace:
    base = {
        "id": 1,
        "daily_quota": None,
        "monthly_quota": None,
        "rpm_limit": None,
        "model_whitelist": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


async def test_daily_quota_exceeded_raises_rate_limit(mock_redis):
    key = _api_key(daily_quota=2)
    with pytest.raises(GovernanceError) as exc:
        for _ in range(3):
            await compatible_governance_service.precheck(
                mock_redis, key, "gpt-4o", "chat/completions"
            )
    assert exc.value.status_code == 429
    assert exc.value.error_type == "rate_limit_error"
    all_keys = [k async for k in mock_redis.scan_iter(match="*")]
    daily_key = next(k for k in all_keys if "daily" in k)
    assert int(await mock_redis.get(daily_key)) == 3


async def test_monthly_quota_exceeded_raises_rate_limit(mock_redis):
    key = _api_key(monthly_quota=1)
    with pytest.raises(GovernanceError) as exc:
        for _ in range(2):
            await compatible_governance_service.precheck(
                mock_redis, key, "gpt-4o", "chat/completions"
            )
    assert exc.value.status_code == 429
    assert exc.value.error_type == "rate_limit_error"


async def test_rpm_limit_exceeded_raises_rate_limit(mock_redis):
    key = _api_key(rpm_limit=1)
    with pytest.raises(GovernanceError) as exc:
        for _ in range(2):
            await compatible_governance_service.precheck(
                mock_redis, key, "gpt-4o", "chat/completions"
            )
    assert exc.value.status_code == 429
    assert exc.value.error_type == "rate_limit_error"


async def test_quota_null_not_limited(mock_redis):
    key = _api_key()
    await compatible_governance_service.precheck(mock_redis, key, "gpt-4o", "chat/completions")
    assert not [k async for k in mock_redis.scan_iter(match="*")]


async def test_quota_zero_not_limited(mock_redis):
    key = _api_key(daily_quota=0, monthly_quota=0, rpm_limit=0)
    await compatible_governance_service.precheck(mock_redis, key, "gpt-4o", "chat/completions")
    assert not [k async for k in mock_redis.scan_iter(match="*")]


async def test_quota_set_expire_on_first_incr(mock_redis):
    key = _api_key(daily_quota=100)
    await compatible_governance_service.precheck(mock_redis, key, "gpt-4o", "chat/completions")
    all_keys = [k async for k in mock_redis.scan_iter(match="*")]
    daily_key = next(k for k in all_keys if "daily" in k)
    assert await mock_redis.ttl(daily_key) == 48 * 3600


async def test_whitelist_model_not_allowed_raises_permission(mock_redis):
    key = _api_key(model_whitelist=["gpt-4o"])
    with pytest.raises(GovernanceError) as exc:
        await compatible_governance_service.precheck(
            mock_redis, key, "claude-3-5", "chat/completions"
        )
    assert exc.value.status_code == 403
    assert exc.value.error_type == "permission_error"


async def test_whitelist_model_allowed_passes(mock_redis):
    key = _api_key(model_whitelist=["gpt-4o"])
    await compatible_governance_service.precheck(mock_redis, key, "gpt-4o", "chat/completions")


async def test_whitelist_none_not_block(mock_redis):
    key = _api_key()
    await compatible_governance_service.precheck(mock_redis, key, "any-model", "chat/completions")


async def test_whitelist_empty_not_block(mock_redis):
    key = _api_key(model_whitelist=[])
    await compatible_governance_service.precheck(mock_redis, key, "any-model", "chat/completions")


async def test_model_none_skips_whitelist(mock_redis):
    key = _api_key(model_whitelist=["gpt-4o"])
    await compatible_governance_service.precheck(mock_redis, key, None, "chat/completions")


async def test_check_model_allowed_second_validation(mock_redis):
    key = _api_key(model_whitelist=["gpt-4o"])
    await compatible_governance_service.check_model_allowed(None, key, "gpt-4o")
    with pytest.raises(GovernanceError) as exc:
        await compatible_governance_service.check_model_allowed(None, key, "deepseek")
    assert exc.value.error_type == "permission_error"


async def test_filter_models_none_key_no_filter(mock_redis):
    models = ["gpt-4o", "deepseek"]
    assert await compatible_governance_service.filter_models(None, None, models) == models


async def test_filter_models_none_whitelist_no_filter(mock_redis):
    models = ["gpt-4o", "deepseek"]
    result = await compatible_governance_service.filter_models(None, _api_key(), models)
    assert result == models


async def test_filter_models_filters_by_whitelist(mock_redis):
    key = _api_key(model_whitelist=["gpt-4o"])
    models = ["gpt-4o", "deepseek"]
    assert await compatible_governance_service.filter_models(None, key, models) == ["gpt-4o"]
    entities = [SimpleNamespace(model_id="gpt-4o"), SimpleNamespace(model_id="deepseek")]
    assert await compatible_governance_service.filter_models(None, key, entities) == [entities[0]]


class _WhitelistRepo:
    def __init__(self, enabled):
        self.enabled = enabled

    async def list_enabled_by_model_id(self, db, model_id):
        return self.enabled.get(model_id, [])


async def test_create_api_key_passes_new_fields(db):
    svc = aks.ApiKeyService(ai_model_repository=_WhitelistRepo({"gpt-4o": ["m1"]}))
    result = await svc.create_api_key(
        db,
        7,
        "测试Key",
        daily_quota=100,
        monthly_quota=2000,
        rpm_limit=30,
        model_whitelist=["gpt-4o"],
    )
    assert result["dailyQuota"] == 100
    assert result["monthlyQuota"] == 2000
    assert result["rpmLimit"] == 30
    assert result["modelWhitelist"] == ["gpt-4o"]


async def test_create_api_key_zero_means_unlimited(db):
    svc = aks.ApiKeyService(ai_model_repository=_WhitelistRepo({"gpt-4o": ["m1"]}))
    result = await svc.create_api_key(db, 7, "无限制", daily_quota=0)
    assert result["dailyQuota"] is None


async def test_create_api_key_invalid_model_raises(db):
    svc = aks.ApiKeyService(ai_model_repository=_WhitelistRepo({"gpt-4o": ["m1"]}))
    with pytest.raises(BusinessException) as exc:
        await svc.create_api_key(
            db=db, user_id=7, name="非法", model_whitelist=["bad-model"]
        )
    assert exc.value.code.code == "A0400"
