from typing import Optional

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

    async def get_by_payment_no(self, db: AsyncSession, payment_no: str) -> Optional[SysPaymentRecord]:
        stmt = select(SysPaymentRecord).where(SysPaymentRecord.payment_no == payment_no)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()


payment_record_repository = PaymentRecordRepository()
