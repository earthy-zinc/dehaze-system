from datetime import datetime
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
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

    async def upsert_by_user_and_package(
        self,
        db: AsyncSession,
        user_id: int,
        package_id: int,
        pay_method: str,
        status: int,
        next_renew_time: Optional[datetime] = None,
        fail_count: int = 0,
    ) -> int:
        """upsert 自动续费配置：冲突时复活（重置 deleted=0, status）"""
        stmt = mysql_insert(SysAutoRenew).values(
            user_id=user_id,
            package_id=package_id,
            pay_method=pay_method,
            status=status,
            next_renew_time=next_renew_time,
            fail_count=fail_count,
        )
        stmt = stmt.on_duplicate_key_update(
            deleted=0,
            status=status,
            pay_method=pay_method,
            next_renew_time=next_renew_time,
            fail_count=fail_count,
            update_time=func.now(),
        )
        await db.execute(stmt)
        # on_duplicate_key_update 不回填 id，需重查
        result = await db.execute(
            select(SysAutoRenew).where(
                SysAutoRenew.user_id == user_id,
                SysAutoRenew.package_id == package_id,
                SysAutoRenew.deleted == 0,
            )
        )
        row = result.scalar_one_or_none()
        return row.id if row else 0

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
