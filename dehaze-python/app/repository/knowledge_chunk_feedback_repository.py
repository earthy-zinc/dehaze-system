from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_knowledge_chunk import SysKnowledgeChunk
from app.models.entity.sys_knowledge_chunk_feedback import SysKnowledgeChunkFeedback
from app.repository.base import BaseRepository


class KnowledgeChunkFeedbackRepository(BaseRepository[SysKnowledgeChunkFeedback]):
    model = SysKnowledgeChunkFeedback

    async def upsert_feedback(
        self,
        db: AsyncSession,
        chunk_id: int,
        user_id: int,
        rating: int,
        comment: str | None,
    ) -> None:
        """记录用户对片段的点赞/点踩：同用户同片段已存在则覆盖，否则插入（幂等）。

        低质量计数 = 该片段 rating=-1 的记录条数，故覆盖（含点赞→点踩/撤销）时计数实时变化。
        """
        stmt = select(SysKnowledgeChunkFeedback).where(
            SysKnowledgeChunkFeedback.chunk_id == chunk_id,
            SysKnowledgeChunkFeedback.user_id == user_id,
        )
        feedback = (await db.execute(stmt)).scalar_one_or_none()
        if feedback:
            feedback.rating = rating
            feedback.comment = comment
            await db.flush()
            return
        await self.create(
            db,
            SysKnowledgeChunkFeedback(
                chunk_id=chunk_id, user_id=user_id, rating=rating, comment=comment
            ),
        )

    async def list_low_quality_by_kb(
        self,
        db: AsyncSession,
        knowledge_base_id: int,
        page: int,
        size: int,
    ) -> tuple[list[dict], int]:
        """按知识库查被点踩片段（thumbs_down_count > 0，降序），关联分块取 content/document_id。

        Returns:
            (rows, total)：rows 为 {chunk_id, content, document_id, thumbs_down_count}
        """
        count_stmt = (
            select(func.count(func.distinct(SysKnowledgeChunk.id)))
            .select_from(SysKnowledgeChunkFeedback)
            .join(SysKnowledgeChunk, SysKnowledgeChunk.id == SysKnowledgeChunkFeedback.chunk_id)
            .where(
                SysKnowledgeChunk.knowledge_base_id == knowledge_base_id,
                SysKnowledgeChunkFeedback.rating == -1,
            )
        )
        total = (await db.execute(count_stmt)).scalar() or 0

        stmt = (
            select(
                SysKnowledgeChunk.id,
                SysKnowledgeChunk.content,
                SysKnowledgeChunk.document_id,
                func.count(SysKnowledgeChunkFeedback.id),
            )
            .join(SysKnowledgeChunk, SysKnowledgeChunk.id == SysKnowledgeChunkFeedback.chunk_id)
            .where(
                SysKnowledgeChunk.knowledge_base_id == knowledge_base_id,
                SysKnowledgeChunkFeedback.rating == -1,
            )
            .group_by(SysKnowledgeChunk.id, SysKnowledgeChunk.content, SysKnowledgeChunk.document_id)
            .order_by(func.count(SysKnowledgeChunkFeedback.id).desc(), SysKnowledgeChunk.id)
            .offset((page - 1) * size)
            .limit(size)
        )
        rows = (await db.execute(stmt)).all()
        return [
            {
                "chunk_id": r[0],
                "content": r[1],
                "document_id": r[2],
                "thumbs_down_count": int(r[3]),
            }
            for r in rows
        ], int(total)


knowledge_chunk_feedback_repository = KnowledgeChunkFeedbackRepository()
