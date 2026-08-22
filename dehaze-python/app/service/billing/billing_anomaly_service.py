"""异常检测与告警服务"""

import logging
from datetime import datetime, timedelta, timezone

from app.config import settings
from app.dependencies.redis import get_redis_client
from app.models.entity.sys_ai_billing import SysAiBilling

logger = logging.getLogger(__name__)

# Redis 异常告警计数 key：ai:anomaly:count:{rule}:{userId}
ANOMALY_COUNT_PREFIX = "ai:anomaly:count:"
# 突发峰值 5 分钟窗口消耗累计 key
BURST_WINDOW_PREFIX = "ai:anomaly:burst:{user_id}:{window}"
BURST_WINDOW_SECONDS = settings.AI_BILLING_ANOMALY_BURST_WINDOW_MINUTES * 60
# 连续配额不足计数 key（24h 内）
QUOTA_FAIL_PREFIX = "ai:anomaly:quota-fail:{userId}"
QUOTA_FAIL_TTL = 24 * 3600

_SHANGHAI_TZ = timezone(timedelta(hours=8))


class BillingAnomalyService:
    """异常检测与告警"""

    @staticmethod
    async def check(
        user_id: int,
        billing_record: SysAiBilling,
        *,
        monthly_limit: int = 0,
        daily_limit: int = 0,
    ) -> None:
        """检查异常规则并告警

        monthly_limit / daily_limit 由调用方（实扣结算）传入，为 0 时跳过限额相关规则。
        redis 经 get_redis_client() 自取，获取失败记日志跳过（不阻断主流程）。
        """
        try:
            redis = await get_redis_client()
        except Exception as e:  # Redis 不可用时异常检测整体跳过
            logger.warning("异常检测 Redis 不可用，跳过 user_id=%s: %s", user_id, e)
            return
        try:
            await _check_single_high(redis, user_id, billing_record, monthly_limit)
            await _check_burst_peak(redis, user_id, billing_record, daily_limit)
            await _check_empty_reply_high_cost(redis, user_id, billing_record)
        except Exception as e:  # 异常检测失败不阻断主流程
            logger.warning("异常检测执行失败 user_id=%s: %s", user_id, e)

    @staticmethod
    async def record_quota_fail(user_id: int) -> None:
        """记录一次配额不足，达到阈值触发告警"""
        try:
            redis = await get_redis_client()
        except Exception as e:
            logger.warning("配额不足计数 Redis 不可用，跳过 user_id=%s: %s", user_id, e)
            return
        key = f"{QUOTA_FAIL_PREFIX}{user_id}"
        try:
            count = await redis.incr(key)
            if count == 1:
                await redis.expire(key, QUOTA_FAIL_TTL)
            if count >= settings.AI_BILLING_ANOMALY_CONSECUTIVE_QUOTA_FAIL:
                await _alert(
                    redis,
                    user_id,
                    "consecutive_quota_fail",
                    f"24h 内配额不足触发 {count} 次，疑似异常使用",
                )
        except Exception as e:
            logger.warning("配额不足异常计数失败 user_id=%s: %s", user_id, e)


async def _check_single_high(
    redis, user_id: int, record: SysAiBilling, monthly_limit: int
) -> None:
    """单次超高：credits > 月限额 × 阈值"""
    if monthly_limit <= 0:
        return
    threshold = monthly_limit * settings.AI_BILLING_ANOMALY_SINGLE_HIGH_THRESHOLD
    if record.credits > threshold:
        await _alert(
            redis,
            user_id,
            "single_high",
            f"单次消耗 {record.credits} 超过月限额 {monthly_limit} 的 10%",
        )


async def _check_burst_peak(
    redis, user_id: int, record: SysAiBilling, daily_limit: int
) -> None:
    """突发峰值：5 分钟窗口内累计消耗 > 日限额 × 阈值"""
    if daily_limit <= 0:
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
            redis,
            user_id,
            "burst_peak",
            (
                f"{settings.AI_BILLING_ANOMALY_BURST_WINDOW_MINUTES} 分钟内消耗 {total} "
                f"超过日限额 {daily_limit} 的 50%"
            ),
        )


async def _check_empty_reply_high_cost(redis, user_id: int, record: SysAiBilling) -> None:
    """空回复高耗：输出 0 token 但输入 > 10000 token"""
    if record.output_tokens == 0 and record.input_tokens > 10000:
        await _alert(
            redis,
            user_id,
            "empty_reply_high_cost",
            f"空回复但输入 {record.input_tokens} token，疑似无效 prompt",
        )


async def _alert(redis, user_id: int, rule_type: str, message: str) -> None:
    """告警：Redis incr 计数 + 日志标记"""
    key = f"{ANOMALY_COUNT_PREFIX}{rule_type}:{user_id}"
    await redis.incr(key)
    await redis.expire(key, 24 * 3600)
    logger.warning("AI计费异常[%s] user_id=%s: %s", rule_type, user_id, message)
