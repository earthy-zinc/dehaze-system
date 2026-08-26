"""AI计费异常检测与告警服务"""

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.dependencies.redis import get_redis_client
from app.models.entity.sys_ai_billing import SysAiBilling
from app.models.schema.ai_billing import (
    AnomalyRecordQuery,
    AnomalyRecordResult,
    AnomalyTrendResult,
)
from app.models.schema.common import PageResult
from app.repository.ai_billing_anomaly_repository import (
    ai_billing_anomaly_repository,
)

logger = logging.getLogger(__name__)

# Redis 异常告警计数 key：ai:anomaly:count:{rule}:{userId}
ANOMALY_COUNT_PREFIX = "ai:anomaly:count:"
# 突发峰值 5 分钟窗口消耗累计 key
BURST_WINDOW_PREFIX = "ai:anomaly:burst:{user_id}:{window}"
BURST_WINDOW_SECONDS = settings.AI_BILLING_ANOMALY_BURST_WINDOW_MINUTES * 60
# 连续配额不足计数 key（24h 内）
QUOTA_FAIL_PREFIX = "ai:anomaly:quota-fail:{userId}"
QUOTA_FAIL_TTL = 24 * 3600

# 异常事件业务时间统一按 Asia/Shanghai（与配额重置/告警窗口时区一致）
_SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


class BillingAnomalyService:
    """异常检测与告警（检测失败不阻断计费主流程）"""

    def __init__(self, ai_billing_anomaly_repository=ai_billing_anomaly_repository):
        self._anomaly_repository = ai_billing_anomaly_repository

    async def check(
        self,
        db: AsyncSession,
        user_id: int,
        billing_record: SysAiBilling,
        *,
        monthly_limit: int = 0,
        daily_limit: int = 0,
    ) -> None:
        """结算后检查四类异常规则并告警落库

        monthly_limit / daily_limit 由调用方（实扣结算）传入，为 0 时跳过限额相关规则。
        Redis 不可用时跳过依赖 Redis 累计的规则（突发峰值），单次超高与空回复高耗仍判定并落库。
        """
        redis = await _try_get_redis(user_id)
        try:
            await _check_single_high(db, redis, user_id, billing_record, monthly_limit, self._anomaly_repository)
            await _check_burst_peak(db, redis, user_id, billing_record, daily_limit, self._anomaly_repository)
            await _check_empty_high_output(db, redis, user_id, billing_record, self._anomaly_repository)
        except Exception as e:  # 异常检测失败不阻断主流程
            logger.warning("异常检测执行失败 user_id=%s: %s", user_id, e)

    async def record_quota_fail(self, db: AsyncSession, user_id: int) -> None:
        """记录一次配额不足，达到阈值触发告警并落库（Redis 不可用时跳过计数）"""
        redis = await _try_get_redis(user_id)
        if redis is None:
            return
        key = f"{QUOTA_FAIL_PREFIX}{user_id}"
        try:
            count = await redis.incr(key)
            if count == 1:
                await redis.expire(key, QUOTA_FAIL_TTL)
            if count >= settings.AI_BILLING_ANOMALY_CONSECUTIVE_QUOTA_FAIL:
                await _alert(
                    db,
                    redis,
                    user_id,
                    None,
                    "consecutive_quota_fail",
                    f"24h 内配额不足触发 {count} 次，疑似异常使用",
                    self._anomaly_repository,
                )
        except Exception as e:
            logger.warning("配额不足异常计数失败 user_id=%s: %s", user_id, e)

    async def list_anomalies(self, db: AsyncSession, query: AnomalyRecordQuery) -> PageResult[AnomalyRecordResult]:
        """异常清单分页查询（管理员 ai:billing:stat）"""
        items, total = await self._anomaly_repository.list_page(
            db,
            query.page,
            query.size,
            user_id=query.user_id,
            anomaly_type=query.anomaly_type,
            status=query.status,
            date_start=query.date_start,
            date_end=query.date_end,
        )
        return PageResult[AnomalyRecordResult](
            list=[AnomalyRecordResult.model_validate(item) for item in items],
            total=total,
        )

    async def anomaly_trend(
        self,
        db: AsyncSession,
        date_start: datetime | None = None,
        date_end: datetime | None = None,
    ) -> list[AnomalyTrendResult]:
        """异常趋势：按类型聚合计数"""
        rows = await self._anomaly_repository.count_group_by_type(
            db, date_start=date_start, date_end=date_end
        )
        return [AnomalyTrendResult(anomaly_type=rule, count=cnt) for rule, cnt in rows]


async def _try_get_redis(user_id: int):
    """获取 Redis 客户端，不可用记日志返回 None（调用方降级处理）"""
    try:
        return await get_redis_client()
    except Exception as e:
        logger.warning("Redis 不可用 user_id=%s: %s", user_id, e)
        return None


async def _alert(
    db: AsyncSession,
    redis,
    user_id: int,
    billing_id: int | None,
    rule_type: str,
    message: str,
    repository,
) -> None:
    """告警：先落库异常事件，再 Redis incr 计数（Redis 不可用不影响落库）"""
    try:
        await repository.create_anomaly(
            db,
            user_id=user_id,
            billing_id=billing_id,
            anomaly_type=rule_type,
            detail=message,
            trigger_at=datetime.now(_SHANGHAI_TZ).replace(tzinfo=None),
        )
    except Exception as e:  # 落库失败仅记日志，不阻断告警
        logger.warning("异常事件落库失败 user_id=%s: %s", user_id, e)
    if redis is None:
        return
    key = f"{ANOMALY_COUNT_PREFIX}{rule_type}:{user_id}"
    await redis.incr(key)
    await redis.expire(key, 24 * 3600)
    logger.warning("AI计费异常[%s] user_id=%s: %s", rule_type, user_id, message)


async def _check_single_high(db, redis, user_id, record, monthly_limit, repository) -> None:
    """单次超高：credits > 月限额 × 阈值"""
    if monthly_limit <= 0:
        return
    threshold = monthly_limit * settings.AI_BILLING_ANOMALY_SINGLE_HIGH_THRESHOLD
    if record.credits > threshold:
        await _alert(
            db,
            redis,
            user_id,
            record.id,
            "single_high",
            f"单次消耗 {record.credits} 超过月限额 {monthly_limit} 的 10%",
            repository,
        )


async def _check_burst_peak(db, redis, user_id, record, daily_limit, repository) -> None:
    """突发峰值：5 分钟窗口内累计消耗 > 日限额 × 阈值（依赖 Redis 累计）"""
    if daily_limit <= 0 or redis is None:
        return
    now = datetime.now(_SHANGHAI_TZ)
    window = int(now.timestamp() // BURST_WINDOW_SECONDS)
    key = f"{BURST_WINDOW_PREFIX.format(user_id=user_id, window=window)}"
    total = await redis.incrby(key, record.credits)
    if total == record.credits:
        await redis.expire(key, BURST_WINDOW_SECONDS * 2)
    threshold = daily_limit * settings.AI_BILLING_ANOMALY_BURST_THRESHOLD
    if total > threshold:
        await _alert(
            db,
            redis,
            user_id,
            record.id,
            "burst",
            (
                f"{settings.AI_BILLING_ANOMALY_BURST_WINDOW_MINUTES} 分钟内消耗 {total} "
                f"超过日限额 {daily_limit} 的 50%"
            ),
            repository,
        )


async def _check_empty_high_output(db, redis, user_id, record, repository) -> None:
    """空回复高耗：输出 0 token 但输入 > 10000 token"""
    if record.output_tokens == 0 and record.input_tokens > 10000:
        await _alert(
            db,
            redis,
            user_id,
            record.id,
            "empty_high_output",
            f"空回复但输入 {record.input_tokens} token，疑似无效 prompt",
            repository,
        )


billing_anomaly_service = BillingAnomalyService()
