"""billing_anomaly_service 单元测试：异常落库、清单分页、趋势聚合、Redis 降级"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

pytestmark = pytest.mark.requires_db

from app.models.entity.sys_ai_billing_anomaly import SysAiBillingAnomaly
from app.models.schema.ai_billing import AnomalyRecordQuery
from app.repository.ai_billing_anomaly_repository import ai_billing_anomaly_repository
from app.service.billing import billing_anomaly_service as am


def _record(credits=0, input_tokens=0, output_tokens=0, billing_id=1):
    return SimpleNamespace(
        id=billing_id, credits=credits,
        input_tokens=input_tokens, output_tokens=output_tokens,
    )


@pytest.fixture
def anomaly_service():
    return am.BillingAnomalyService(ai_billing_anomaly_repository=ai_billing_anomaly_repository)


async def _all_anomalies(db):
    return (await db.execute(select(SysAiBillingAnomaly))).scalars().all()


class TestAnomalyPersistence:
    async def test_four_rules_persist(self, db, mock_redis, anomaly_service):
        await anomaly_service.check(
            db, 1, _record(credits=15000, billing_id=1), monthly_limit=100000
        )
        await anomaly_service.check(
            db, 1, _record(credits=6000, billing_id=2), daily_limit=10000
        )
        await anomaly_service.check(
            db, 1, _record(input_tokens=15000, output_tokens=0, billing_id=3)
        )
        for _ in range(10):
            await anomaly_service.record_quota_fail(db, 1)

        rows = await _all_anomalies(db)
        assert sorted(r.anomaly_type for r in rows) == [
            "burst", "consecutive_quota_fail", "empty_high_output", "single_high",
        ]
        by_type = {r.anomaly_type: r for r in rows}
        assert by_type["single_high"].billing_id == 1
        assert by_type["burst"].billing_id == 2
        assert by_type["empty_high_output"].billing_id == 3
        assert by_type["consecutive_quota_fail"].billing_id is None
        assert all(r.status == 0 for r in rows)
        assert all(r.trigger_at is not None for r in rows)

    async def test_alert_failure_does_not_block_redis_count(self, db, mock_redis):
        class _BrokenRepo:
            async def create_anomaly(self, *args, **kwargs):
                raise RuntimeError("db down")

        svc = am.BillingAnomalyService(ai_billing_anomaly_repository=_BrokenRepo())
        await svc.check(db, 1, _record(credits=15000), monthly_limit=100000)
        assert int(await mock_redis.get("ai:anomaly:count:single_high:1")) == 1
        assert await _all_anomalies(db) == []

    async def test_redis_unavailable_still_persists(self, db, monkeypatch, anomaly_service):
        async def _broken():
            raise ConnectionError("redis down")

        monkeypatch.setattr(am, "get_redis_client", _broken)
        await anomaly_service.check(
            db, 1, _record(credits=999999, billing_id=9), monthly_limit=1
        )
        await anomaly_service.record_quota_fail(db, 1)

        rows = await _all_anomalies(db)
        assert len(rows) == 1
        assert rows[0].anomaly_type == "single_high"
        assert rows[0].billing_id == 9


class TestAnomalyQuery:
    @staticmethod
    async def _seed(db):
        now = datetime.now()
        seeds = [
            (1, "single_high", now),
            (1, "burst", now - timedelta(days=1)),
            (2, "empty_high_output", now - timedelta(days=2)),
        ]
        for i, (uid, typ, trigger_at) in enumerate(seeds):
            db.add(SysAiBillingAnomaly(
                user_id=uid, billing_id=i + 1, anomaly_type=typ,
                detail="测试记录", trigger_at=trigger_at,
            ))
        await db.flush()

    async def test_list_anomalies_filters_and_pages(self, db, anomaly_service):
        await self._seed(db)

        result = await anomaly_service.list_anomalies(
            db, AnomalyRecordQuery(user_id=1, page=1, size=10)
        )
        assert result.total == 2
        assert {r.anomaly_type for r in result.list} == {"single_high", "burst"}

        result = await anomaly_service.list_anomalies(
            db, AnomalyRecordQuery(anomaly_type="burst", page=1, size=10)
        )
        assert result.total == 1
        assert result.list[0].user_id == 1

        result = await anomaly_service.list_anomalies(
            db, AnomalyRecordQuery(status=0, page=1, size=1)
        )
        assert result.total == 3
        assert len(result.list) == 1

    async def test_list_anomalies_filters_by_time_range(self, db, anomaly_service):
        await self._seed(db)
        now = datetime.now()

        result = await anomaly_service.list_anomalies(
            db, AnomalyRecordQuery(date_start=now - timedelta(hours=1), page=1, size=10)
        )
        assert result.total == 1
        assert result.list[0].anomaly_type == "single_high"

        result = await anomaly_service.list_anomalies(
            db, AnomalyRecordQuery(date_end=now - timedelta(hours=1), page=1, size=10)
        )
        assert result.total == 2

    async def test_anomaly_trend_groups_by_type(self, db, anomaly_service):
        await self._seed(db)
        trend = await anomaly_service.anomaly_trend(db)
        mapping = {t.anomaly_type: t.count for t in trend}
        assert mapping == {"single_high": 1, "burst": 1, "empty_high_output": 1}
