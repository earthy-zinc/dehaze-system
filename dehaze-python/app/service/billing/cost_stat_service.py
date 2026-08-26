"""成本-利润统计服务（毛利核算双口径）"""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.entity.sys_ai_billing import SysAiBilling
from app.models.entity.sys_order import SysOrder
from app.models.schema.ai_billing_cost import CostStatResult

# 已实收订单状态（已支付/已完成）
_PAID_ORDER_STATUSES = (2, 3)


class CostStatService:
    """毛利核算双口径：

    - overall（整体毛利官方口径）：income = 全部订单实收（积分卡 + 会员卡），cost = Σ sys_ai_billing.cost
    - ai（AI 参考口径）：aiIncome = 积分卡实收 + 会员卡实收 × AI 分摊比例（ai.billing.membership-ai-ratio）

    不按"积分消耗 × 兑换率"核算收入；积分卡量价优惠/赠送作为营销成本。成本数据仅管理员可见。
    """

    async def cost_stats(
        self,
        db: AsyncSession,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[CostStatResult]:
        cost_stmt = select(func.coalesce(func.sum(SysAiBilling.cost), 0))
        if start_time:
            cost_stmt = cost_stmt.where(SysAiBilling.create_time >= start_time)
        if end_time:
            cost_stmt = cost_stmt.where(SysAiBilling.create_time <= end_time)
        total_cost = Decimal((await db.execute(cost_stmt)).scalar() or 0)

        income_stmt = (
            select(
                SysOrder.package_type,
                func.coalesce(func.sum(SysOrder.paid_amount), 0),
            )
            .where(SysOrder.status.in_(_PAID_ORDER_STATUSES))
            .group_by(SysOrder.package_type)
        )
        if start_time:
            income_stmt = income_stmt.where(SysOrder.paid_time >= start_time)
        if end_time:
            income_stmt = income_stmt.where(SysOrder.paid_time <= end_time)
        rows = (await db.execute(income_stmt)).all()

        credit_income = Decimal("0")
        vip_income = Decimal("0")
        for package_type, amount in rows:
            revenue = Decimal(amount) / 100  # 分 -> 元
            if package_type == "credit":
                credit_income += revenue
            else:
                vip_income += revenue

        overall_income = credit_income + vip_income
        ai_income = credit_income + vip_income * Decimal(
            str(settings.AI_BILLING_MEMBERSHIP_AI_RATIO)
        )
        return [
            self._build("overall", overall_income, total_cost),
            self._build("ai", ai_income, total_cost),
        ]

    @staticmethod
    def _build(metric: str, revenue: Decimal, cost: Decimal) -> CostStatResult:
        profit = revenue - cost
        profit_rate = float(profit / revenue) if revenue else 0.0
        return CostStatResult(
            metric=metric,
            revenue=float(revenue.quantize(Decimal("0.01"))),
            cost=float(cost.quantize(Decimal("0.01"))),
            profit=float(profit.quantize(Decimal("0.01"))),
            profit_rate=round(profit_rate, 4),
        )


cost_stat_service = CostStatService()
