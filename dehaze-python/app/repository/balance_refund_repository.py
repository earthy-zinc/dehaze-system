from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_balance_refund import SysBalanceRefund
from app.repository.base import BaseRepository


class BalanceRefundRepository(BaseRepository[SysBalanceRefund]):
    model = SysBalanceRefund

    async def get_by_refund_no(
        self, db: AsyncSession, refund_no: str
    ) -> SysBalanceRefund | None:
        stmt = select(SysBalanceRefund).where(SysBalanceRefund.refund_no == refund_no)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_pending(self, db: AsyncSession) -> list[SysBalanceRefund]:
        stmt = (
            select(SysBalanceRefund)
            .where(SysBalanceRefund.deleted == 0, SysBalanceRefund.status == 1)
            .order_by(SysBalanceRefund.id.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())


balance_refund_repository = BalanceRefundRepository()
