from datetime import datetime
from typing import Optional

from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_message import SysMessage
from app.repository.base import BaseRepository, escape_like


class MessageRepository(BaseRepository[SysMessage]):
    model = SysMessage

    async def get_page(
        self,
        db: AsyncSession,
        recipient_id: int,
        page: int,
        page_size: int,
        type: Optional[str] = None,
        read_status: Optional[int] = None,
    ) -> tuple[list[SysMessage], int]:
        stmt = select(SysMessage).where(
            SysMessage.recipient_id == recipient_id,
            SysMessage.deleted == 0,
        )
        if type:
            stmt = stmt.where(SysMessage.type == type)
        if read_status is not None:
            stmt = stmt.where(SysMessage.read_status == read_status)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        stmt = stmt.order_by(SysMessage.read_status.asc(), SysMessage.create_time.desc(), SysMessage.id.desc())
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(stmt)
        items = list(result.scalars().all())
        return items, total

    async def search(
        self,
        db: AsyncSession,
        recipient_id: int,
        keyword: str,
        page: int,
        page_size: int,
    ) -> tuple[list[SysMessage], int]:
        escaped = escape_like(keyword)
        like_pattern = f"%{escaped}%"
        stmt = select(SysMessage).where(
            SysMessage.recipient_id == recipient_id,
            SysMessage.deleted == 0,
            (SysMessage.title.like(like_pattern, escape="\\"))
            | (SysMessage.content.like(like_pattern, escape="\\")),
        )

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        stmt = stmt.order_by(SysMessage.read_status.asc(), SysMessage.create_time.desc(), SysMessage.id.desc())
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(stmt)
        items = list(result.scalars().all())
        return items, total

    async def count_unread(self, db: AsyncSession, recipient_id: int) -> int:
        stmt = select(func.count()).where(
            SysMessage.recipient_id == recipient_id,
            SysMessage.read_status == 0,
            SysMessage.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar() or 0

    async def get_by_id_and_recipient(
        self,
        db: AsyncSession,
        message_id: int,
        recipient_id: int,
    ) -> Optional[SysMessage]:
        stmt = select(SysMessage).where(
            SysMessage.id == message_id,
            SysMessage.recipient_id == recipient_id,
            SysMessage.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def mark_read(self, db: AsyncSession, message_id: int, recipient_id: int) -> bool:
        stmt = (
            update(SysMessage)
            .where(
                SysMessage.id == message_id,
                SysMessage.recipient_id == recipient_id,
                SysMessage.read_status == 0,
                SysMessage.deleted == 0,
            )
            .values(read_status=1, read_time=datetime.now())
        )
        result = await db.execute(stmt)
        return result.rowcount > 0

    async def mark_all_read(
        self,
        db: AsyncSession,
        recipient_id: int,
        type: Optional[str] = None,
    ) -> int:
        stmt = (
            update(SysMessage)
            .where(
                SysMessage.recipient_id == recipient_id,
                SysMessage.read_status == 0,
                SysMessage.deleted == 0,
            )
            .values(read_status=1, read_time=datetime.now())
        )
        if type:
            stmt = stmt.where(SysMessage.type == type)
        result = await db.execute(stmt)
        return result.rowcount

    async def soft_delete_by_ids_and_recipient(
        self,
        db: AsyncSession,
        ids: list[int],
        recipient_id: int,
    ) -> int:
        stmt = (
            update(SysMessage)
            .where(
                SysMessage.id.in_(ids),
                SysMessage.recipient_id == recipient_id,
            )
            .values(deleted=1)
        )
        result = await db.execute(stmt)
        return result.rowcount

    async def find_by_biz(
        self,
        db: AsyncSession,
        biz_module: str,
        biz_id: str,
    ) -> list[SysMessage]:
        stmt = select(SysMessage).where(
            SysMessage.biz_module == biz_module,
            SysMessage.biz_id == biz_id,
            SysMessage.deleted == 0,
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def batch_create(self, db: AsyncSession, messages: list[SysMessage]) -> list[SysMessage]:
        db.add_all(messages)
        await db.flush()
        return messages

    async def delete_expired(self, db: AsyncSession, now: datetime, batch_size: int = 500) -> int:
        total_deleted = 0
        while True:
            id_stmt = select(SysMessage.id).where(
                SysMessage.expires_at < now,
            ).limit(batch_size)
            id_result = await db.execute(id_stmt)
            ids = [row[0] for row in id_result.fetchall()]
            if not ids:
                break
            stmt = delete(SysMessage).where(SysMessage.id.in_(ids))
            result = await db.execute(stmt)
            total_deleted += result.rowcount
            if len(ids) < batch_size:
                break
        return total_deleted


message_repository = MessageRepository()
