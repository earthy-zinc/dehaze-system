"""权益配置域：等级权益列表（带缓存）与配置修改。"""

import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.redis import get_redis_client
from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback
from app.repository.member_benefit_repository import member_benefit_repository
from app.service.member.member_service import _benefit_to_vo, _invalidate_member_cache

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
        keys = []
        async for key in redis.scan_iter("member:benefit-summary:*", count=100):
            keys.append(key)
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
                pass

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

        await db.flush()
        await _invalidate_member_cache(level_code=level_code)
        await _invalidate_benefit_summary_cache()


member_benefit_service = MemberBenefitService()
