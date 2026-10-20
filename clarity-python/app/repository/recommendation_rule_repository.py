"""
推荐规则数据访问层
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_recommendation_rule import SysRecommendationRule
from app.repository.base import BaseRepository


class RecommendationRuleRepository(BaseRepository[SysRecommendationRule]):
    model = SysRecommendationRule

    async def get_enabled_rules(self, db: AsyncSession) -> list[SysRecommendationRule]:
        stmt = (
            select(SysRecommendationRule)
            .where(
                SysRecommendationRule.enabled == 1,
                SysRecommendationRule.deleted == 0,
            )
            .order_by(SysRecommendationRule.weight.desc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_all_rules(self, db: AsyncSession) -> list[SysRecommendationRule]:
        stmt = (
            select(SysRecommendationRule)
            .where(SysRecommendationRule.deleted == 0)
            .order_by(SysRecommendationRule.weight.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())


recommendation_rule_repository = RecommendationRuleRepository()
