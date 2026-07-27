from datetime import datetime
from typing import Optional

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_announcement import SysAnnouncement
from app.repository.base import BaseRepository, escape_like


class AnnouncementRepository(BaseRepository[SysAnnouncement]):
    model = SysAnnouncement

    async def get_page(
        self,
        db: AsyncSession,
        page: int,
        page_size: int,
        title: Optional[str] = None,
        type: Optional[str] = None,
        status: Optional[int] = None,
    ) -> tuple[list[SysAnnouncement], int]:
        stmt = select(SysAnnouncement).where(SysAnnouncement.deleted == 0)
        if title:
            stmt = stmt.where(
                SysAnnouncement.title.like(f"%{escape_like(title)}%", escape="\\")
            )
        if type:
            stmt = stmt.where(SysAnnouncement.type == type)
        if status is not None:
            stmt = stmt.where(SysAnnouncement.status == status)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        stmt = stmt.order_by(SysAnnouncement.create_time.desc(), SysAnnouncement.id.desc())
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(stmt)
        items = list(result.scalars().all())
        return items, total

    async def get_by_id(
        self,
        db: AsyncSession,
        announcement_id: int,
    ) -> Optional[SysAnnouncement]:
        stmt = select(SysAnnouncement).where(
            SysAnnouncement.id == announcement_id,
            SysAnnouncement.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def soft_delete(self, db: AsyncSession, announcement_id: int) -> bool:
        stmt = (
            update(SysAnnouncement)
            .where(
                SysAnnouncement.id == announcement_id,
                SysAnnouncement.deleted == 0,
            )
            .values(deleted=1)
        )
        result = await db.execute(stmt)
        return result.rowcount > 0

    async def update_status(
        self,
        db: AsyncSession,
        announcement_id: int,
        status: int,
        sent_count: Optional[int] = None,
        send_time: Optional[datetime] = None,
    ) -> bool:
        values: dict = {"status": status}
        if sent_count is not None:
            values["sent_count"] = sent_count
        if send_time is not None:
            values["send_time"] = send_time
        stmt = (
            update(SysAnnouncement)
            .where(
                SysAnnouncement.id == announcement_id,
                SysAnnouncement.deleted == 0,
            )
            .values(**values)
        )
        result = await db.execute(stmt)
        return result.rowcount > 0

    async def get_scheduled_pending(
        self,
        db: AsyncSession,
        now: datetime,
    ) -> list[SysAnnouncement]:
        stmt = select(SysAnnouncement).where(
            SysAnnouncement.status == 2,
            SysAnnouncement.deleted == 0,
            SysAnnouncement.send_time <= now,
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())


announcement_repository = AnnouncementRepository()
