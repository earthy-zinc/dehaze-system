from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_payment_record import SysPaymentRecord
from app.repository.base import BaseRepository


class PaymentRecordRepository(BaseRepository[SysPaymentRecord]):
    model = SysPaymentRecord

    async def list_by_order_id(self, db: AsyncSession, order_id: int) -> list[SysPaymentRecord]:
        stmt = (
            select(SysPaymentRecord)
            .where(SysPaymentRecord.order_id == order_id)
            .order_by(SysPaymentRecord.id.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_by_payment_no(self, db: AsyncSession, payment_no: str) -> SysPaymentRecord | None:
        stmt = select(SysPaymentRecord).where(SysPaymentRecord.payment_no == payment_no)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_pending_by_order_id(self, db: AsyncSession, order_id: int) -> SysPaymentRecord | None:
        """支付阶段预写的处理中流水（status=1），回调成功时原地更新。"""
        stmt = select(SysPaymentRecord).where(
            SysPaymentRecord.order_id == order_id,
            SysPaymentRecord.status == 1,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_success_between(
        self, db: AsyncSession, start, end
    ) -> list[SysPaymentRecord]:
        """指定时间段内支付成功的流水（对账数据源，按回调时间归日）。"""
        stmt = select(SysPaymentRecord).where(
            SysPaymentRecord.status == 2,
            SysPaymentRecord.callback_time >= start,
            SysPaymentRecord.callback_time < end,
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())


payment_record_repository = PaymentRecordRepository()
