from datetime import datetime
from typing import Optional

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_package import SysPackage
from app.repository.base import BaseRepository, escape_like


class PackageRepository(BaseRepository[SysPackage]):
    model = SysPackage

    async def get_by_name(self, db: AsyncSession, name: str) -> Optional[SysPackage]:
        stmt = select(SysPackage).where(
            SysPackage.name == name,
            SysPackage.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_on_sale(self, db: AsyncSession) -> list[SysPackage]:
        stmt = (
            select(SysPackage)
            .where(SysPackage.deleted == 0, SysPackage.status == 1)
            .order_by(SysPackage.sort.asc(), SysPackage.id.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_page(
        self,
        db: AsyncSession,
        page: int,
        page_size: int,
        *,
        name: Optional[str] = None,
        level_code: Optional[str] = None,
        period: Optional[str] = None,
        status: Optional[int] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> tuple[list[SysPackage], int]:
        stmt = select(SysPackage).where(SysPackage.deleted == 0)

        if name:
            escaped = escape_like(name)
            stmt = stmt.where(SysPackage.name.like(f"%{escaped}%", escape="\\"))
        if level_code:
            stmt = stmt.where(SysPackage.level_code == level_code)
        if period:
            stmt = stmt.where(SysPackage.period == period)
        if status is not None:
            stmt = stmt.where(SysPackage.status == status)
        if start_time:
            stmt = stmt.where(SysPackage.create_time >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"))
        if end_time:
            stmt = stmt.where(SysPackage.create_time <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S"))

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        stmt = stmt.order_by(SysPackage.sort.asc(), SysPackage.id.desc())
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(stmt)
        items = list(result.scalars().all())
        return items, total

    async def increment_sales_count(self, db: AsyncSession, package_id: int, count: int = 1) -> None:
        stmt = (
            update(SysPackage)
            .where(SysPackage.id == package_id)
            .values(sales_count=SysPackage.sales_count + count)
        )
        await db.execute(stmt)
        await db.flush()


package_repository = PackageRepository()
