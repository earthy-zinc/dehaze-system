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

    async def latest_status_by_billing_ids(
        self,
        db: AsyncSession,
        billing_ids: list[int],
    ) -> dict[int, int]:
        """批量查询计费记录的最新退款状态（0:无;1:待审核;2:已通过;3:已驳回）"""
        if not billing_ids:
            return {}
        stmt = (
            select(SysAiRefund.billing_id, SysAiRefund.status)
            .where(SysAiRefund.billing_id.in_(billing_ids))
            .order_by(SysAiRefund.id.asc())
        )
        result = await db.execute(stmt)
        status_map: dict[int, int] = {}
        for billing_id, status in result.all():
            status_map[billing_id] = status
        return status_map


ai_refund_repository = AiRefundRepository()
