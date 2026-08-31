from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_recharge import SysRecharge
from app.repository.base import BaseRepository


class RechargeRepository(BaseRepository[SysRecharge]):
    model = SysRecharge

    async def get_by_recharge_no(
        self, db: AsyncSession, recharge_no: str
    ) -> SysRecharge | None:
        stmt = select(SysRecharge).where(SysRecharge.recharge_no == recharge_no)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()


recharge_repository = RechargeRepository()
