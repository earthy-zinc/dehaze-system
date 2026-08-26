"""月结账单服务"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.redis import get_redis_client
from app.infrastructure.cache.cache import CacheService
from app.models.schema.ai_billing import BillResult
from app.repository.ai_billing_repository import ai_billing_repository
from app.repository.ai_credit_log_repository import ai_credit_log_repository

logger = logging.getLogger(__name__)

_TZ = ZoneInfo("Asia/Shanghai")
BILL_CACHE_PREFIX = "ai:bill:{user_id}:{month}"
# 消费与充值退款 source 白名单
_RECHARGE_SOURCES = {"recharge", "vip_gift", "trial", "admin_adjust", "vip_gift_expire"}


def _month_bounds(month: str) -> tuple[datetime, datetime]:
    """解析 YYYY-MM，返回 (月初, 月末 23:59:59)，非法格式抛参数异常"""
    try:
        start = datetime.strptime(month, "%Y-%m").replace(tzinfo=_TZ)
    except ValueError:
        raise BusinessException(ResultCode.PARAM_ERROR, "月份格式不正确，应为 YYYY-MM") from None
    end = (start + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
    return start, end


def _is_empty_bill(bill: BillResult, month: str) -> bool:
    """账单是否为空账期（无任何消费/充值/退款记录）。"""
    now = datetime.now(_TZ)
    if month == now.strftime("%Y-%m"):
        # 当前月份允许返回全 0（月初尚无数据属正常）
        return False
    return bill.total_consume == 0 and bill.total_recharge == 0 and bill.total_refund == 0


class BillService:
    """月结账单"""

    def __init__(
        self,
        ai_billing_repository=ai_billing_repository,
        ai_credit_log_repository=ai_credit_log_repository,
    ):
        self.ai_billing_repository = ai_billing_repository
        self.ai_credit_log_repository = ai_credit_log_repository

    async def generate_monthly_bill(self, db: AsyncSession, user_id: int, month: str) -> BillResult:
        """生成月结账单并缓存到 Redis（幂等，可重新生成）"""
        redis = await get_redis_client()
        month_start, month_end = _month_bounds(month)

        # 1. 汇总消耗：按 bill_type 分组
        by_type = await self.ai_billing_repository.sum_credits_by_user_group_by_bill_type(
            db, user_id, month_start, month_end
        )
        item_summary = {row["bill_type"]: row["credits"] for row in by_type}
        total_consume = sum(item_summary.values())

        # 2. 汇总充值/退款：按 source 分组
        by_source = await self.ai_credit_log_repository.sum_amount_by_user_and_source(
            db, user_id, month_start, month_end
        )
        recharge_amounts = [amount for src, amount in by_source.items() if src in _RECHARGE_SOURCES]
        total_recharge = int(sum(recharge_amounts))
        total_refund = int(by_source.get("refund", Decimal(0)))

        # 3. 余额变动：月初（上月末）→ 月末
        balance_start = await self.ai_credit_log_repository.get_balance_at_or_before(
            db, user_id, month_start - timedelta(seconds=1)
        )
        balance_end = await self.ai_credit_log_repository.get_balance_at_or_before(
            db, user_id, month_end
        )

        bill = BillResult(
            user_id=user_id,
            month=month,
            total_consume=total_consume,
            total_recharge=total_recharge,
            total_refund=total_refund,
            balance_start=balance_start,
            balance_end=balance_end,
            item_summary=item_summary,
        )
        await CacheService(redis).set_json(
            BILL_CACHE_PREFIX.format(user_id=user_id, month=month),
            bill.model_dump(mode="json"),
            settings.AI_BILLING_BILL_CACHE_TTL,
        )
        return bill

    async def get_bill(self, db: AsyncSession, user_id: int, month: str) -> BillResult:
        """查询月结账单（Redis 优先，未命中重新生成）

        非当前月份且该月无任何消费/充值/退款记录时视为账单不存在。
        """
        redis = await get_redis_client()
        _month_bounds(month)  # 提前校验月份格式
        cache = CacheService(redis)
        cache_key = BILL_CACHE_PREFIX.format(user_id=user_id, month=month)
        cached = await cache.get_json(cache_key)
        if cached is not None:
            bill = BillResult.model_validate(cached)
            if _is_empty_bill(bill, month):
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "账单不存在")
            return bill

        bill = await self.generate_monthly_bill(db, user_id, month)
        # 非当前月份且无任何消费/充值/退款记录：视为账单不存在。
        # 空账期不入缓存，否则后续查询命中全 0 缓存会错误地返回成功而非 A0401。
        if _is_empty_bill(bill, month):
            await cache.delete(cache_key)
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "账单不存在")
        return bill


bill_service = BillService()
