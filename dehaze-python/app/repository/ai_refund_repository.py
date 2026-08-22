from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_refund import SysAiRefund
from app.repository.base import BaseRepository


class AiRefundRepository(BaseRepository[SysAiRefund]):
    model = SysAiRefund

    async def create_refund(
        self,
        db: AsyncSession,
        **kwargs: Any,
    ) -> SysAiRefund:
        refund = SysAiRefund(**kwargs)
        return await self.create(db, refund)

    async def get_pending_by_billing_id(
        self,
        db: AsyncSession,
        billing_id: int,
    ) -> SysAiRefund | None:
        """按计费记录查询未完结的退款申请（用于不重复申请校验）"""
        stmt = (
            select(SysAiRefund)
            .where(
                SysAiRefund.billing_id == billing_id,
                SysAiRefund.status == 1,  # 待审核
            )
            .limit(1)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_status(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        *,
        status: int | None = None,
    ) -> tuple[list[SysAiRefund], int]:
        stmt = select(SysAiRefund)
        if status:
            stmt = stmt.where(SysAiRefund.status == status)
        stmt = stmt.order_by(SysAiRefund.create_time.desc(), SysAiRefund.id.desc())
        return await self.paginate(db, stmt, page, size)


ai_refund_repository = AiRefundRepository()
