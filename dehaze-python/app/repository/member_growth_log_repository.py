from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_member_growth_log import SysMemberGrowthLog
from app.repository.base import BaseRepository


class MemberGrowthLogRepository(BaseRepository[SysMemberGrowthLog]):
    model = SysMemberGrowthLog

    async def get_page(
        self,
        db: AsyncSession,
        user_id: int,
        page: int,
        page_size: int,
        *,
        change_type: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> tuple[list[SysMemberGrowthLog], int]:
        stmt = select(SysMemberGrowthLog).where(SysMemberGrowthLog.user_id == user_id)

        if change_type:
            stmt = stmt.where(SysMemberGrowthLog.change_type == change_type)
        if start_time:
            start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            stmt = stmt.where(SysMemberGrowthLog.create_time >= start_dt)
        if end_time:
            end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            stmt = stmt.where(SysMemberGrowthLog.create_time <= end_dt)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        stmt = stmt.order_by(SysMemberGrowthLog.create_time.desc(), SysMemberGrowthLog.id.desc())
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(stmt)
        items = list(result.scalars().all())
        return items, total

    async def create_log(
        self,
        db: AsyncSession,
        user_id: int,
        change_type: str,
        change_value: int,
        balance: int,
        related_id: str | None = None,
        reason: str | None = None,
        operator_id: int | None = None,
    ) -> SysMemberGrowthLog:
        log = SysMemberGrowthLog(
            user_id=user_id,
            change_type=change_type,
            change_value=change_value,
            balance=balance,
            related_id=related_id,
            reason=reason,
            operator_id=operator_id,
        )
        db.add(log)
        await db.flush()
        await db.refresh(log)
        return log


member_growth_log_repository = MemberGrowthLogRepository()
