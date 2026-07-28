from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_promotion import SysPromotion, SysPromotionPackage
from app.repository.base import BaseRepository


class PromotionRepository(BaseRepository[SysPromotion]):
    model = SysPromotion

    async def list_active(self, db: AsyncSession) -> list[SysPromotion]:
        now = datetime.now()
        stmt = (
            select(SysPromotion)
            .where(
                SysPromotion.deleted == 0,
                SysPromotion.status == 1,
                SysPromotion.start_time <= now,
                SysPromotion.end_time >= now,
            )
            .order_by(SysPromotion.id.desc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

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


promotion_repository = PromotionRepository()
