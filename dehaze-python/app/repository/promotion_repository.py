from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_promotion import SysPromotion, SysPromotionPackage
from app.repository.base import BaseRepository, escape_like


class PromotionRepository(BaseRepository[SysPromotion]):
    model = SysPromotion

    async def list_active_by_package_id(self, db: AsyncSession, package_id: int) -> list[dict]:
        now = datetime.now()
        stmt = (
            select(SysPromotion, SysPromotionPackage)
            .join(
                SysPromotionPackage,
                SysPromotionPackage.promotion_id == SysPromotion.id,
            )
            .where(
                SysPromotion.deleted == 0,
                SysPromotion.status == 1,
                SysPromotion.start_time <= now,
                SysPromotion.end_time >= now,
                SysPromotionPackage.package_id == package_id,
            )
        )
        result = await db.execute(stmt)
        rows = result.all()
        return [
            {
                "promotion": row[0],
                "promotion_package": row[1],
            }
            for row in rows
        ]

    async def get_page(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        name: str | None = None,
        type: str | None = None,
        status: int | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> tuple[list[SysPromotion], int]:
        stmt = select(SysPromotion).where(SysPromotion.deleted == 0)
        if name:
            stmt = stmt.where(SysPromotion.name.like(f"%{escape_like(name)}%", escape="\\"))
        if type:
            stmt = stmt.where(SysPromotion.type == type)
        if status is not None:
            stmt = stmt.where(SysPromotion.status == status)
        if start_time:
            stmt = stmt.where(SysPromotion.start_time >= start_time)
        if end_time:
            stmt = stmt.where(SysPromotion.end_time <= end_time)
        stmt = stmt.order_by(SysPromotion.id.desc())
        return await self.paginate(db, stmt, page, size)

    async def create(self, db: AsyncSession, promotion: SysPromotion) -> SysPromotion:
        return await super().create(db, promotion)

    async def update(self, db: AsyncSession, promotion_id: int, data: dict) -> None:
        promotion = await self.get_by_id(db, promotion_id)
        if promotion:
            for key, value in data.items():
                setattr(promotion, key, value)
            await db.flush()

    async def soft_delete(self, db: AsyncSession, promotion_id: int) -> None:
        await self.soft_delete_by_ids(db, [promotion_id])

    async def delete_packages_by_promotion(
        self, db: AsyncSession, promotion_id: int
    ) -> None:
        stmt = select(SysPromotionPackage).where(
            SysPromotionPackage.promotion_id == promotion_id
        )
        result = await db.execute(stmt)
        for row in result.scalars().all():
            await db.delete(row)
        await db.flush()

    async def list_package_ids_by_promotion(
        self, db: AsyncSession, promotion_id: int
    ) -> list[int]:
        stmt = select(SysPromotionPackage.package_id).where(
            SysPromotionPackage.promotion_id == promotion_id
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def bind_packages(
        self,
        db: AsyncSession,
        promotion_id: int,
        packages: list[SysPromotionPackage],
    ) -> None:
        del_stmt = (
            select(SysPromotionPackage)
            .where(SysPromotionPackage.promotion_id == promotion_id)
        )
        result = await db.execute(del_stmt)
        for row in result.scalars().all():
            await db.delete(row)
        if packages:
            db.add_all(packages)
        await db.flush()


promotion_repository = PromotionRepository()
