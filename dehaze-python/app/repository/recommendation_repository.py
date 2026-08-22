"""
推荐记录数据访问层
"""

from datetime import datetime

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_recommendation import SysRecommendation
from app.repository.base import BaseRepository


class RecommendationRepository(BaseRepository[SysRecommendation]):
    model = SysRecommendation

    async def count_total(
        self, db: AsyncSession, start: datetime | None, end: datetime | None
    ) -> int:
        stmt = select(func.count()).select_from(SysRecommendation)
        if start:
            stmt = stmt.where(SysRecommendation.create_time >= start)
        if end:
            stmt = stmt.where(SysRecommendation.create_time <= end)
        return (await db.execute(stmt)).scalar() or 0

    async def count_useful(
        self, db: AsyncSession, start: datetime | None, end: datetime | None
    ) -> int:
        stmt = select(func.count()).where(SysRecommendation.feedback == 1)
        if start:
            stmt = stmt.where(SysRecommendation.create_time >= start)
        if end:
            stmt = stmt.where(SysRecommendation.create_time <= end)
        return (await db.execute(stmt)).scalar() or 0

    async def count_feedback_total(
        self, db: AsyncSession, start: datetime | None, end: datetime | None
    ) -> int:
        stmt = select(func.count()).where(SysRecommendation.feedback.in_([1, 2]))
        if start:
            stmt = stmt.where(SysRecommendation.create_time >= start)
        if end:
            stmt = stmt.where(SysRecommendation.create_time <= end)
        return (await db.execute(stmt)).scalar() or 0

    async def count_adopted_algorithm_distinct(
        self, db: AsyncSession, start: datetime | None, end: datetime | None
    ) -> int:
        stmt = select(func.count(func.distinct(SysRecommendation.adopted_algorithm_id))).where(
            SysRecommendation.adopted_algorithm_id.isnot(None)
        )
        if start:
            stmt = stmt.where(SysRecommendation.create_time >= start)
        if end:
            stmt = stmt.where(SysRecommendation.create_time <= end)
        return (await db.execute(stmt)).scalar() or 0

    async def select_daily_adoption_rate(
        self, db: AsyncSession, start: datetime | None, end: datetime | None
    ) -> list[dict]:
        subq = select(
            func.date(SysRecommendation.create_time).label("date"),
            func.count().label("total"),
            func.sum(case((SysRecommendation.feedback == 1, 1), else_=0)).label("useful"),
        ).where(SysRecommendation.feedback.in_([1, 2]))

        if start:
            subq = subq.where(SysRecommendation.create_time >= start)
        if end:
            subq = subq.where(SysRecommendation.create_time <= end)

        subq = subq.group_by(func.date(SysRecommendation.create_time)).subquery()

        stmt = select(
            subq.c.date,
            (func.coalesce(subq.c.useful, 0) * 1.0 / func.nullif(subq.c.total, 0)).label(
                "adoptionRate"
            ),
        ).order_by(subq.c.date.asc())

        result = await db.execute(stmt)
        return [{"date": str(row[0]), "adoptionRate": float(row[1] or 0)} for row in result.all()]

    async def get_latest_by_image_md5(
        self, db: AsyncSession, image_md5: str
    ) -> SysRecommendation | None:
        stmt = (
            select(SysRecommendation)
            .where(SysRecommendation.image_md5 == image_md5)
            .order_by(SysRecommendation.id.desc())
            .limit(1)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()


recommendation_repository = RecommendationRepository()
