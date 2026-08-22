"""充值与赠送服务"""

from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.service.billing.balance_service import balance_service


class RechargeService:
    """充值与赠送"""

    async def recharge(self, 
        db: AsyncSession,
        user_id: int,
        amount: int,
        source: str,
        related_id: int | None = None,
        reason: str | None = None,
        operator_id: int | None = None,
    ) -> Decimal:
        """充值/赠送/试用/管理员调整

        source: recharge(积分包购买) / vip_gift(VIP赠送) / trial(试用) / admin_adjust(管理员调整)
        余额增加（Redis INCR + MySQL CAS + 清欠费 + 写流水），返回变动后余额。
        """
        return await balance_service.increase(
            db,
            user_id,
            int(amount),
            source=source,
            related_id=related_id,
            reason=reason,
            operator_id=operator_id,
        )

    async def grant_trial_credits(self, db: AsyncSession, user_id: int) -> Decimal:
        """新用户注册赠送试用积分"""
        return await self.recharge(
            db,
            user_id,
            settings.AI_BILLING_TRIAL_CREDITS,
            source="trial",
            reason="新用户注册试用赠送",
        )

    async def grant_vip_monthly_gift(self, 
        db: AsyncSession,
        user_id: int,
        amount: int,
    ) -> Decimal:
        """VIP 按月赠送积分"""
        return await self.recharge(
            db,
            user_id,
            amount,
            source="vip_gift",
            reason="VIP 按月赠送",
        )


recharge_service = RechargeService()
