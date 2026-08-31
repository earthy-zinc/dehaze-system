from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_reconciliation import SysReconciliation
from app.repository.base import BaseRepository


class ReconciliationRepository(BaseRepository[SysReconciliation]):
    model = SysReconciliation

    async def delete_by_date(self, db: AsyncSession, recon_date) -> None:
        """重跑对账时清除同日差异记录（差异以当日全量重算为准）。"""
        stmt = delete(SysReconciliation).where(SysReconciliation.recon_date == recon_date)
        await db.execute(stmt)

    async def list_by_date(self, db: AsyncSession, recon_date) -> list[SysReconciliation]:
        stmt = (
            select(SysReconciliation)
            .where(SysReconciliation.deleted == 0, SysReconciliation.recon_date == recon_date)
            .order_by(SysReconciliation.id.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())


reconciliation_repository = ReconciliationRepository()
