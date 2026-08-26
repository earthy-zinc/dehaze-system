"""AI 供应商/模型运营统计服务（管理端运营视图，GET /api/v1/ai/usage/stats）

数据口径（最简可靠，与文档 §4 一致）：
- providerHealth：逐供应商读 Redis 健康快照（provider_health_service.get_health_snapshot）
- modelUsage：sys_ai_billing 按实际模型聚合（调用数/Token/积分）
- degradeFault.downgradeFrequency：sys_ai_billing 中 actual_model 非空按原选模型聚合（发生降级）
- degradeFault.keyFailoverCount：Redis 中处于冷却期（临时不可用）的 Key 数（近期失败切换的当前快照）
"""

from datetime import datetime

from redis.asyncio import Redis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.provider.provider_health_service import provider_health_service
from app.infrastructure.provider.provider_key_selector import KEY_UNAVAILABLE_PREFIX
from app.models.entity.sys_ai_billing import SysAiBilling
from app.models.entity.sys_ai_model import SysAiModel
from app.models.schema.ai_provider import (
    DegradeFaultStatResult,
    DowngradeStatResult,
    ModelUsageStatResult,
    ProviderHealthStatResult,
    UsageStatsQuery,
    UsageStatsResult,
)
from app.repository.ai_billing_repository import ai_billing_repository
from app.repository.ai_model_repository import ai_model_repository
from app.repository.ai_provider_repository import ai_provider_repository


class AiUsageStatsService:
    """运营统计聚合：供应商健康 + 模型用量 + 降级/Key 故障"""

    def __init__(
        self,
        ai_provider_repository=ai_provider_repository,
        ai_model_repository=ai_model_repository,
        ai_billing_repository=ai_billing_repository,
        provider_health_service=provider_health_service,
    ):
        self.ai_provider_repository = ai_provider_repository
        self.ai_model_repository = ai_model_repository
        self.ai_billing_repository = ai_billing_repository
        self.provider_health_service = provider_health_service

    async def get_usage_stats(
        self, db: AsyncSession, redis: Redis, query: UsageStatsQuery
    ) -> UsageStatsResult:
        provider_health = await self._provider_health(db, redis)
        model_usage = await self._model_usage(db, query)
        degrade_fault = await self._degrade_fault(db, redis, query)
        return UsageStatsResult(
            provider_health=provider_health,
            model_usage=model_usage,
            degrade_fault=degrade_fault,
        )

    async def _provider_health(
        self, db: AsyncSession, redis: Redis
    ) -> list[ProviderHealthStatResult]:
        providers = await self.ai_provider_repository.get_all(db)
        items = []
        for p in providers:
            snapshot = await self.provider_health_service.get_health_snapshot(redis, p.id)
            items.append(
                ProviderHealthStatResult(
                    provider_id=p.id,
                    provider_name=p.display_name,
                    health=snapshot["status"],
                    call_count=snapshot["total_calls_24h"],
                    success_rate=snapshot["success_rate"],
                    rate429=snapshot["limit_rate"],
                    p95_latency_ms=snapshot["p95_latency_ms"],
                    circuit_open=snapshot["circuit_open"],
                )
            )
        return items

    async def _model_usage(self, db: AsyncSession, query: UsageStatsQuery) -> list[ModelUsageStatResult]:
        stmt = (
            select(
                SysAiBilling.model,
                func.count(SysAiBilling.id),
                func.coalesce(func.sum(SysAiBilling.input_tokens), 0),
                func.coalesce(func.sum(SysAiBilling.output_tokens), 0),
                func.coalesce(func.sum(SysAiBilling.credits), 0),
            )
            .group_by(SysAiBilling.model)
        )
        if query.start_time:
            stmt = stmt.where(SysAiBilling.create_time >= query.start_time)
        if query.end_time:
            stmt = stmt.where(SysAiBilling.create_time <= query.end_time)
        rows = (await db.execute(stmt)).all()
        model_ids = [r[0] for r in rows]
        display_names = await self._model_display_names(db, model_ids)
        return [
            ModelUsageStatResult(
                model_id=r[0],
                display_name=display_names.get(r[0], r[0]),
                call_count=int(r[1]),
                input_tokens=int(r[2]),
                output_tokens=int(r[3]),
                credits=int(r[4]),
            )
            for r in rows
        ]

    async def _degrade_fault(
        self, db: AsyncSession, redis: Redis, query: UsageStatsQuery
    ) -> DegradeFaultStatResult:
        downgrade_rows = await self._degradation_by_model(db, query)
        downgrade_frequency = [
            DowngradeStatResult(model_id=r["dimension"], count=r["degradation_count"])
            for r in downgrade_rows
            if r["degradation_count"] > 0
        ]
        key_failover_count = await self._count_unavailable_keys(redis)
        return DegradeFaultStatResult(
            downgrade_frequency=downgrade_frequency,
            key_failover_count=key_failover_count,
        )

    async def _degradation_by_model(self, db: AsyncSession, query: UsageStatsQuery) -> list[dict]:
        stmt = (
            select(
                SysAiBilling.actual_model,
                func.count(SysAiBilling.id),
            )
            .where(SysAiBilling.actual_model.isnot(None))
            .group_by(SysAiBilling.actual_model)
        )
        if query.start_time:
            stmt = stmt.where(SysAiBilling.create_time >= query.start_time)
        if query.end_time:
            stmt = stmt.where(SysAiBilling.create_time <= query.end_time)
        rows = (await db.execute(stmt)).all()
        return [{"dimension": str(r[0]), "degradation_count": int(r[1])} for r in rows]

    async def _count_unavailable_keys(self, redis: Redis) -> int:
        count = 0
        async for _ in redis.scan_iter(match=KEY_UNAVAILABLE_PREFIX.format("*")):
            count += 1
        return count

    async def _model_display_names(self, db: AsyncSession, model_ids: list[str]) -> dict[str, str]:
        if not model_ids:
            return {}
        stmt = select(SysAiModel.model_id, SysAiModel.display_name).where(
            SysAiModel.model_id.in_(model_ids)
        )
        rows = (await db.execute(stmt)).all()
        return {model_id: display_name for model_id, display_name in rows}


ai_usage_stats_service = AiUsageStatsService()
