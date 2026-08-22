from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_agent_thought import SysAiAgentThought
from app.repository.base import BaseRepository


class AiAgentThoughtRepository(BaseRepository[SysAiAgentThought]):
    model = SysAiAgentThought

    async def list_by_message(
        self,
        db: AsyncSession,
        message_id: int,
    ) -> list[SysAiAgentThought]:
        stmt = select(SysAiAgentThought).where(SysAiAgentThought.message_id == message_id)
        stmt = stmt.order_by(SysAiAgentThought.position.asc())
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def create_thought(
        self,
        db: AsyncSession,
        **kwargs: Any,
    ) -> SysAiAgentThought:
        thought = SysAiAgentThought(**kwargs)
        return await self.create(db, thought)


ai_agent_thought_repository = AiAgentThoughtRepository()
