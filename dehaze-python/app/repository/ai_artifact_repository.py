from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_artifact import SysAiArtifact
from app.repository.base import BaseRepository


def _order_newest_first(stmt):
    return stmt.order_by(SysAiArtifact.create_time.desc(), SysAiArtifact.id.desc())


class AiArtifactRepository(BaseRepository[SysAiArtifact]):
    model = SysAiArtifact

    async def list_by_conversation(
        self,
        db: AsyncSession,
        conv_id: int,
        page: int,
        size: int,
    ) -> tuple[list[SysAiArtifact], int]:
        stmt = select(SysAiArtifact).where(
            SysAiArtifact.conversation_id == conv_id,
        )
        stmt = _order_newest_first(stmt)
        return await self.paginate(db, stmt, page, size)

    async def list_by_message(
        self,
        db: AsyncSession,
        message_id: int,
    ) -> list[SysAiArtifact]:
        stmt = select(SysAiArtifact).where(
            SysAiArtifact.message_id == message_id,
        )
        stmt = _order_newest_first(stmt)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def list_by_message_ids(
        self,
        db: AsyncSession,
        message_ids: list[int],
    ) -> list[SysAiArtifact]:
        """批量查询多消息关联的产物（过滤失效，供上下文组装引用层）"""
        if not message_ids:
            return []
        stmt = select(SysAiArtifact).where(
            SysAiArtifact.message_id.in_(message_ids),
            SysAiArtifact.is_invalid == 0,
        )
        stmt = _order_newest_first(stmt)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def list_by_ref(
        self,
        db: AsyncSession,
        ref_type: str,
        ref_id: int,
    ) -> list[SysAiArtifact]:
        """按业务引用反查产物列表（同一业务记录可能产生多条产物）"""
        stmt = select(SysAiArtifact).where(
            SysAiArtifact.ref_type == ref_type,
            SysAiArtifact.ref_id == ref_id,
        )
        stmt = _order_newest_first(stmt)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_by_ref(
        self,
        db: AsyncSession,
        ref_type: str,
        ref_id: int,
    ) -> SysAiArtifact | None:
        stmt = select(SysAiArtifact).where(
            SysAiArtifact.ref_type == ref_type,
            SysAiArtifact.ref_id == ref_id,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def mark_invalid(
        self,
        db: AsyncSession,
        ref_type: str,
        ref_id: int,
    ) -> None:
        stmt = (
            update(SysAiArtifact)
            .where(
                SysAiArtifact.ref_type == ref_type,
                SysAiArtifact.ref_id == ref_id,
                SysAiArtifact.is_invalid == 0,
            )
            .values(is_invalid=1)
        )
        await db.execute(stmt)


ai_artifact_repository = AiArtifactRepository()
