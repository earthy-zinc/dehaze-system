"""权益配置域：等级权益列表（带缓存）与配置修改。"""

import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.redis import get_redis_client
from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback
from app.models.entity.sys_member import QUOTA_TASK_TYPES
from app.repository.member_benefit_repository import member_benefit_repository
from app.repository.member_repository import member_repository
from app.repository.package_repository import package_repository
from app.service.member.member_service import _benefit_to_vo, _invalidate_member_cache
from app.service.member.quota_service import _effective_task_quota, resolve_card_overrides

logger = logging.getLogger(__name__)

MEMBER_BENEFIT_CACHE_TTL = 3600

BENEFIT_FIELD_MAP = {
    "levelName": "level_name",
    "growthMin": "growth_min",
    "growthMax": "growth_max",
    "monthlyDehazeQuota": "monthly_dehaze_quota",
    "monthlyDerainQuota": "monthly_derain_quota",
    "monthlyDesnowQuota": "monthly_desnow_quota",
    "monthlyLowlightQuota": "monthly_lowlight_quota",
    "monthlySuperResolutionQuota": "monthly_super_resolution_quota",
    "monthlyDenoiseQuota": "monthly_denoise_quota",
    "monthlyInpaintQuota": "monthly_inpaint_quota",
    "monthlyEvaluateQuota": "monthly_evaluate_quota",
    "aiCreditsDaily": "ai_credits_daily",
    "aiCreditsMonthly": "ai_credits_monthly",
    "multimodalLimit": "multimodal_limit",
    "maxDevices": "max_devices",
    "vipGiftCredits": "vip_gift_credits",
    "historyRetention": "history_retention",
    "batchLimit": "batch_limit",
    "priority": "priority",
    "advancedParams": "advanced_params",
    "hdExport": "hd_export",
    "reportExport": "report_export",
    "batchDownload": "batch_download",
    "sort": "sort",
    "status": "status",
}

# AI 限额字段（非负校验）
AI_LIMIT_FIELDS = (
    "ai_credits_daily",
    "ai_credits_monthly",
    "multimodal_limit",
    "vip_gift_credits",
)


async def _invalidate_benefit_summary_cache() -> None:
    """权益配置修改后失效所有用户的权益概览聚合缓存"""

    async def _scan_delete():
        redis = await get_redis_client()
        keys = [key async for key in redis.scan_iter("member:benefit-summary:*", count=100)]
        if keys:
            await redis.delete(*keys)

    await redis_operation_with_fallback(
        _scan_delete, default=None, operation_name="benefit_summary_cache_invalidate"
    )


class MemberBenefitService:
    def __init__(self, member_benefit_repository=member_benefit_repository):
        self.member_benefit_repository = member_benefit_repository

    async def list_benefits(self, db: AsyncSession) -> list[dict]:
        cache_key = "member:benefit:all"

        async def _get_cache():
            redis = await get_redis_client()
            return await redis.get(cache_key)

        cached_raw = await redis_operation_with_fallback(
            _get_cache, default=None, operation_name="member_benefit_list_cache_get"
        )
        if cached_raw:
            try:
                return json.loads(cached_raw)
            except (json.JSONDecodeError, TypeError):
                # 缓存体损坏：忽略缓存回源 DB 重建（降级），但需可见以暴露缓存被写坏
                logger.warning("权益列表缓存损坏，回退查库重建: key=%s", cache_key, exc_info=True)

        benefits = await self.member_benefit_repository.list_all(db)
        result = [_benefit_to_vo(b) for b in benefits]

        async def _set_cache():
            redis = await get_redis_client()
            await redis.setex(
                cache_key,
                MEMBER_BENEFIT_CACHE_TTL,
                json.dumps(result, ensure_ascii=False, default=str),
            )

        await redis_operation_with_fallback(
            _set_cache, default=None, operation_name="member_benefit_list_cache_set"
        )

        return result

    async def _refresh_level_member_quotas(self, db: AsyncSession, benefit) -> None:
        """权益配置修改立即生效：批量刷新该等级活跃会员的 8 类配额快照（不含已用）
        并失效其配额缓存
        """
        package = await package_repository.get_by_level_code(db, benefit.level_code)

        offset, batch_size = 0, 500
        while True:
            members = await member_repository.list_active_by_level(
                db, benefit.level_code, offset=offset, limit=batch_size
            )
            if not members:
                return

            keys = []
            for m in members:
                effective = _effective_task_quota(benefit, resolve_card_overrides(m, package))
                for task_type in QUOTA_TASK_TYPES:
                    setattr(m, f"monthly_{task_type}_quota", effective[task_type])
                    keys.append(f"member:quota:{m.user_id}:{task_type}")
            await db.flush()

            async def _del(batch_keys: tuple[str, ...] = tuple(keys)):
                redis = await get_redis_client()
                await redis.delete(*batch_keys)

            await redis_operation_with_fallback(
                _del, default=None, operation_name="benefit_update_member_quota_cache"
            )
            offset += batch_size

    async def update_benefit(self, db: AsyncSession, level_code: str, form: dict) -> None:
        benefit = await self.member_benefit_repository.get_by_level_code(db, level_code)
        if not benefit:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "权益配置不存在")

        for camel_key, snake_key in BENEFIT_FIELD_MAP.items():
            if camel_key in form and form[camel_key] is not None:
                setattr(benefit, snake_key, form[camel_key])

        if (
            benefit.growth_max
            and benefit.growth_max > 0
            and benefit.growth_min > benefit.growth_max
        ):
            raise BusinessException(ResultCode.BENEFIT_CONFIG_INVALID, "成长值下限不能大于上限")

        for field in AI_LIMIT_FIELDS:
            if getattr(benefit, field) is not None and getattr(benefit, field) < 0:
                raise BusinessException(ResultCode.BENEFIT_CONFIG_INVALID, "AI 限额字段不能为负数")

        if benefit.max_devices < 1:
            raise BusinessException(
                ResultCode.BENEFIT_CONFIG_INVALID, "同时在线设备数上限不能小于1"
            )

        await db.flush()
        await self._refresh_level_member_quotas(db, benefit)
        await _invalidate_member_cache(level_code=level_code)
        await _invalidate_benefit_summary_cache()


member_benefit_service = MemberBenefitService()
