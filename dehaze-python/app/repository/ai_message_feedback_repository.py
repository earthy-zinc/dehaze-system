from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import get_audit_update_values
from app.models.entity.sys_ai_message_feedback import SysAiMessageFeedback
from app.repository.base import BaseRepository


class AiMessageFeedbackRepository(BaseRepository[SysAiMessageFeedback]):
    model = SysAiMessageFeedback

    async def upsert_feedback(
        self,
        db: AsyncSession,
        message_id: int,
        user_id: int,
        rating: int,
        tags: list[str] | None,
        comment: str | None,
        conversation_id: int | None = None,
        model: str | None = None,
        source: str = "internal",
    ) -> SysAiMessageFeedback:
        """存在则更新并复活（deleted=0），不存在则插入。需绕过软删过滤查全表以命中唯一索引。"""
        stmt = (
            select(SysAiMessageFeedback)
            .where(
                SysAiMessageFeedback.message_id == message_id,
                SysAiMessageFeedback.user_id == user_id,
            )
            .execution_options(include_deleted=True)
        )
        result = await db.execute(stmt)
        feedback = result.scalar_one_or_none()
        if feedback:
            feedback.rating = rating
            feedback.tags = tags
            feedback.comment = comment
            feedback.conversation_id = conversation_id
            feedback.model = model
            feedback.source = source
            feedback.processed = 0
            feedback.deleted = 0
            await db.flush()
            await db.refresh(feedback)
            return feedback
        feedback = SysAiMessageFeedback(
            message_id=message_id,
            user_id=user_id,
            conversation_id=conversation_id,
            model=model,
            source=source,
            rating=rating,
            tags=tags,
            comment=comment,
            processed=0,
        )
        return await self.create(db, feedback)

    async def get_by_user_and_message(
        self,
        db: AsyncSession,
        message_id: int,
        user_id: int,
    ) -> SysAiMessageFeedback | None:
        stmt = select(SysAiMessageFeedback).where(
            SysAiMessageFeedback.message_id == message_id,
            SysAiMessageFeedback.user_id == user_id,
            SysAiMessageFeedback.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def soft_delete(
        self,
        db: AsyncSession,
        message_id: int,
        user_id: int,
    ) -> int:
        values = {"deleted": 1}
        values.update(get_audit_update_values())
        stmt = (
            update(SysAiMessageFeedback)
            .where(
                SysAiMessageFeedback.message_id == message_id,
                SysAiMessageFeedback.user_id == user_id,
                SysAiMessageFeedback.deleted == 0,
            )
            .values(**values)
        )
        result = await db.execute(stmt)
        return result.rowcount


ai_message_feedback_repository = AiMessageFeedbackRepository()
