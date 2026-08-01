from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_member_benefit import SysMemberBenefit
from app.repository.base import BaseRepository


class MemberBenefitRepository(BaseRepository[SysMemberBenefit]):
    model = SysMemberBenefit

    async def get_by_level_code(self, db: AsyncSession, level_code: str) -> Optional[SysMemberBenefit]:
        """根据等级编码查询会员权益（含软删记录，用于查重）"""
        stmt = select(SysMemberBenefit).where(
            SysMemberBenefit.level_code == level_code,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(self, db: AsyncSession) -> list[SysMemberBenefit]:
        stmt = (
            select(SysMemberBenefit)
            .where(SysMemberBenefit.deleted == 0)
            .order_by(SysMemberBenefit.sort.asc(), SysMemberBenefit.id.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def list_ordered_by_growth_min(self, db: AsyncSession) -> list[SysMemberBenefit]:
        stmt = (
            select(SysMemberBenefit)
            .where(SysMemberBenefit.deleted == 0, SysMemberBenefit.status == 1)
            .order_by(SysMemberBenefit.growth_min.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())


member_benefit_repository = MemberBenefitRepository()
