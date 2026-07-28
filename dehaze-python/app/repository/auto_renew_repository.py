from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_auto_renew import SysAutoRenew
from app.repository.base import BaseRepository


class AutoRenewRepository(BaseRepository[SysAutoRenew]):
    model = SysAutoRenew

    async def get_by_user_and_package(
        self,
        db: AsyncSession,
        user_id: int,
        package_id: int,
    ) -> Optional[SysAutoRenew]:
        stmt = select(SysAutoRenew).where(
            SysAutoRenew.user_id == user_id,
            SysAutoRenew.package_id == package_id,
            SysAutoRenew.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_due(self, db: AsyncSession) -> list[SysAutoRenew]:
        stmt = select(SysAutoRenew).where(
            SysAutoRenew.status == 1,
            SysAutoRenew.next_renew_time.isnot(None),
            SysAutoRenew.next_renew_time <= datetime.now(),
            SysAutoRenew.deleted == 0,
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())


auto_renew_repository = AutoRenewRepository()
