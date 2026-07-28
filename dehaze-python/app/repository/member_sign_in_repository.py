from datetime import date
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_member_sign_in import SysMemberSignIn
from app.repository.base import BaseRepository


class MemberSignInRepository(BaseRepository[SysMemberSignIn]):
    model = SysMemberSignIn

    async def get_by_user_and_date(
        self, db: AsyncSession, user_id: int, sign_date: date
    ) -> Optional[SysMemberSignIn]:
        stmt = select(SysMemberSignIn).where(
            SysMemberSignIn.user_id == user_id,
            SysMemberSignIn.sign_date == sign_date,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_user_and_date_range(
        self,
        db: AsyncSession,
        user_id: int,
        start_date: date,
        end_date: date,
    ) -> list[SysMemberSignIn]:
        stmt = (
            select(SysMemberSignIn)
            .where(
                SysMemberSignIn.user_id == user_id,
                SysMemberSignIn.sign_date >= start_date,
                SysMemberSignIn.sign_date <= end_date,
            )
            .order_by(SysMemberSignIn.sign_date.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_latest_continuous_days(
        self, db: AsyncSession, user_id: int, before_date: date
    ) -> int:
        stmt = select(SysMemberSignIn.continuous_days).where(
            SysMemberSignIn.user_id == user_id,
            SysMemberSignIn.sign_date == before_date,
        )
        result = await db.execute(stmt)
        row = result.first()
        return row[0] if row else 0

    async def count_month_sign_in(
        self, db: AsyncSession, user_id: int, start_date: date, end_date: date
    ) -> int:
        stmt = select(func.count()).where(
            SysMemberSignIn.user_id == user_id,
            SysMemberSignIn.sign_date >= start_date,
            SysMemberSignIn.sign_date <= end_date,
        )
        result = await db.execute(stmt)
        return result.scalar() or 0


member_sign_in_repository = MemberSignInRepository()
