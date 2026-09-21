"""Key 级 rpm/日额度限制测试（真实 db fixture + fakeredis）

覆盖 list_usable_keys 分钟限额过滤与取 Key 时的原子「检查+预留」（与文档 §2.3 同源）。
"""

import asyncio
from datetime import datetime

import pytest

from app.infrastructure.provider.provider_key_selector import (
    KEY_DAILY_PREFIX,
    KEY_MINUTE_PREFIX,
    provider_key_selector,
)
from app.models.entity.sys_ai_provider_key import SysAiProviderKey
from app.models.schema.ai_provider import ProviderCreate, ProviderKeyCreate
from app.service.ai_provider_key_service import ai_provider_key_service
from app.service.ai_provider_service import ai_provider_service

pytestmark = pytest.mark.requires_db


async def _create_provider_with_key(db, mock_redis, code: str, **key_overrides) -> tuple[int, int]:
    provider = await ai_provider_service.create_provider(
        db,
        mock_redis,
        ProviderCreate(
            provider_code=code, display_name=f"P-{code}", api_base_url="https://api.test.local/v1"
        ),
    )
    form = ProviderKeyCreate(name="k", key=f"sk-{code}-0123456789", **key_overrides)
    key = await ai_provider_key_service.create_key(db, provider.id, form)
    return provider.id, key.id


async def _usable_key_ids(db, mock_redis, provider_id: int) -> set[int]:
    usable = await provider_key_selector.list_usable_keys(db, mock_redis, provider_id)
    return {k.id for k in usable}


def _now_minute_key(key_id: int) -> str:
    return KEY_MINUTE_PREFIX.format(key_id, datetime.now().strftime("%Y%m%d%H%M"))


def _now_daily_key(key_id: int) -> str:
    return KEY_DAILY_PREFIX.format(key_id, datetime.now().strftime("%Y%m%d"))


def _stub_key(
    key_id: int, daily_quota: int | None = None, rpm_limit: int | None = None
) -> SysAiProviderKey:
    # 用真实实体（SqlAlchemy 模型可脱离 session 实例化），满足 reserve_key 的契约
    return SysAiProviderKey(
        id=key_id, daily_quota=daily_quota, rpm_limit=rpm_limit, priority=0, weight=1
    )


class TestRpmLimit:
    async def test_minute_limit_excludes_key(self, db, mock_redis):
        provider_id, key_id = await _create_provider_with_key(
            db, mock_redis, "rpm_excl", rpm_limit=2
        )
        await mock_redis.set(_now_minute_key(key_id), 2, ex=60)
        assert key_id not in await _usable_key_ids(db, mock_redis, provider_id)

    async def test_minute_limit_below_quota_keeps_key(self, db, mock_redis):
        provider_id, key_id = await _create_provider_with_key(
            db, mock_redis, "rpm_kept", rpm_limit=2
        )
        await mock_redis.set(_now_minute_key(key_id), 1, ex=60)
        assert key_id in await _usable_key_ids(db, mock_redis, provider_id)

    async def test_no_rpm_limit_never_excluded(self, db, mock_redis):
        provider_id, key_id = await _create_provider_with_key(db, mock_redis, "rpm_free")
        await mock_redis.set(_now_minute_key(key_id), 999999, ex=60)
        assert key_id in await _usable_key_ids(db, mock_redis, provider_id)

    async def test_mark_call_success_does_not_double_count(self, db, mock_redis):
        """额度在取 Key 时已原子预留，成功标记不重复计数"""
        provider_id, key_id = await _create_provider_with_key(
            db, mock_redis, "rpm_incr", rpm_limit=10, daily_quota=100
        )
        await provider_key_selector.select_key(db, mock_redis, provider_id)
        await provider_key_selector.mark_call_success(mock_redis, key_id, used_by=1)
        assert int(await mock_redis.get(_now_minute_key(key_id))) == 1
        assert int(await mock_redis.get(_now_daily_key(key_id))) == 1


class TestQuotaReservation:
    async def test_select_key_reserves_quota(self, db, mock_redis):
        provider_id, key_id = await _create_provider_with_key(
            db, mock_redis, "reserve_one", rpm_limit=10, daily_quota=100
        )
        await provider_key_selector.select_key(db, mock_redis, provider_id)
        await provider_key_selector.select_key(db, mock_redis, provider_id)
        assert int(await mock_redis.get(_now_minute_key(key_id))) == 2
        assert int(await mock_redis.get(_now_daily_key(key_id))) == 2

    async def test_concurrent_reserve_never_exceeds_rpm(self, db, mock_redis):
        """并发取 Key：检查与计数必须原子，rpm 不得被击穿"""
        provider_id, key_id = await _create_provider_with_key(
            db, mock_redis, "reserve_race", rpm_limit=5
        )
        keys = await provider_key_selector.list_usable_keys(db, mock_redis, provider_id)
        granted = await asyncio.gather(
            *[provider_key_selector.reserve_key(mock_redis, keys[0]) for _ in range(20)]
        )
        assert sum(granted) == 5
        assert int(await mock_redis.get(_now_minute_key(key_id))) == 5

    async def test_daily_rolled_back_when_minute_limit_hit(self, mock_redis):
        """分钟超限被拒时回滚已加的日计数，失败预留不得占用日额度"""
        key = _stub_key(77, daily_quota=10, rpm_limit=1)
        assert await provider_key_selector.reserve_key(mock_redis, key) is True
        assert await provider_key_selector.reserve_key(mock_redis, key) is False
        assert int(await mock_redis.get(_now_daily_key(77))) == 1
        assert int(await mock_redis.get(_now_minute_key(77))) == 1

    async def test_no_limit_never_rejected(self, mock_redis):
        key = _stub_key(78)
        assert all(
            await asyncio.gather(
                *[provider_key_selector.reserve_key(mock_redis, key) for _ in range(10)]
            )
        )
        assert await mock_redis.get(_now_daily_key(78)) is None

    async def test_daily_quota_exhausted_rejects(self, mock_redis):
        key = _stub_key(79, daily_quota=2)
        assert await provider_key_selector.reserve_key(mock_redis, key) is True
        assert await provider_key_selector.reserve_key(mock_redis, key) is True
        assert await provider_key_selector.reserve_key(mock_redis, key) is False
