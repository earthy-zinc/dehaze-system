from typing import Optional

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_message_template import SysMessageTemplate
from app.repository.base import BaseRepository, escape_like


class MessageTemplateRepository(BaseRepository[SysMessageTemplate]):
    model = SysMessageTemplate

    async def get_page(
        self,
        db: AsyncSession,
        page: int,
        page_size: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        status: Optional[int] = None,
    ) -> tuple[list[SysMessageTemplate], int]:
        stmt = select(SysMessageTemplate).where(SysMessageTemplate.deleted == 0)
        if name:
            stmt = stmt.where(
                SysMessageTemplate.name.like(f"%{escape_like(name)}%", escape="\\")
            )
        if type:
            stmt = stmt.where(SysMessageTemplate.type == type)
        if status is not None:
            stmt = stmt.where(SysMessageTemplate.status == status)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        stmt = stmt.order_by(SysMessageTemplate.id.asc())
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(stmt)
        items = list(result.scalars().all())
        return items, total

    async def get_by_id(self, db: AsyncSession, template_id: int) -> Optional[SysMessageTemplate]:
        stmt = select(SysMessageTemplate).where(
            SysMessageTemplate.id == template_id,
            SysMessageTemplate.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_code(self, db: AsyncSession, code: str) -> Optional[SysMessageTemplate]:
        stmt = select(SysMessageTemplate).where(
            SysMessageTemplate.code == code,
            SysMessageTemplate.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()


message_template_repository = MessageTemplateRepository()
