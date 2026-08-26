"""供应商/Key 新增字段（user_identity_forward / rpm_limit）服务层测试（真实 db fixture）"""

import pytest

from app.models.schema.ai_provider import (
    ProviderCreate,
    ProviderKeyCreate,
    ProviderUpdate,
)
from app.service.ai_provider_key_service import ai_provider_key_service
from app.service.ai_provider_service import ai_provider_service


def _provider_form(**overrides) -> ProviderCreate:
    data = dict(
        provider_code="openai",
        display_name="OpenAI",
        api_base_url="https://api.openai.com/v1",
        protocol_type="openai_compat",
        auth_type="bearer",
        default_headers=None,
        sort_order=0,
        health_check_enabled=1,
        status=1,
    )
    data.update(overrides)
    return ProviderCreate(**data)


async def test_create_provider_persists_user_identity_forward(db, mock_redis):
    form = _provider_form(
        provider_code="prov_fwd",
        user_identity_forward={"enabled": True, "field": "user_id", "prefix": "dh_", "max_len": 64},
    )
    result = await ai_provider_service.create_provider(db, mock_redis, form)
    assert result.user_identity_forward.enabled is True
    assert result.user_identity_forward.field == "user_id"
    assert result.user_identity_forward.prefix == "dh_"
    assert result.user_identity_forward.max_len == 64


async def test_create_provider_without_forward_is_null(db, mock_redis):
    result = await ai_provider_service.create_provider(
        db, mock_redis, _provider_form(provider_code="prov_nofwd")
    )
    assert result.user_identity_forward is None


async def test_update_provider_forward_field(db, mock_redis):
    provider = await ai_provider_service.create_provider(
        db, mock_redis, _provider_form(provider_code="prov_upd")
    )
    updated = await ai_provider_service.update_provider(
        db,
        mock_redis,
        provider.id,
        ProviderUpdate(
            user_identity_forward={
                "enabled": False,
                "field": "metadata.user_id",
                "prefix": "u_",
                "max_len": 128,
            }
        ),
    )
    assert updated.user_identity_forward.enabled is False
    assert updated.user_identity_forward.field == "metadata.user_id"


async def test_create_key_persists_rpm_limit(db, mock_redis):
    provider = await ai_provider_service.create_provider(
        db, mock_redis, _provider_form(provider_code="prov_key_rpm")
    )
    form = ProviderKeyCreate(
        name="主Key",
        key="sk-test-rpm-1234567890",
        priority=0,
        weight=1,
        status=1,
        rpm_limit=60,
    )
    result = await ai_provider_key_service.create_key(db, provider.id, form)
    assert result.rpm_limit == 60
    # Key 列表也返回 rpm_limit
    keys = await ai_provider_key_service.list_keys(db, provider.id)
    assert keys[0].rpm_limit == 60


async def test_create_key_default_rpm_limit_zero(db, mock_redis):
    provider = await ai_provider_service.create_provider(
        db, mock_redis, _provider_form(provider_code="prov_key_rpm0")
    )
    form = ProviderKeyCreate(name="主Key", key="sk-test-rpm-0000", priority=0, weight=1, status=1)
    result = await ai_provider_key_service.create_key(db, provider.id, form)
    assert result.rpm_limit == 0
