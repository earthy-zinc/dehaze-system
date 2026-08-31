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

    async def list_by_messages(
        self,
        db: AsyncSession,
        message_ids: list[int],
    ) -> dict[int, list[SysAiAgentThought]]:
        """批量查询多条消息的推理步骤，按 message_id 分组（position 正序）。

        供会话消息列表批量附带思考链，避免逐条 N+1 查询。
        """
        if not message_ids:
            return {}
        stmt = select(SysAiAgentThought).where(
            SysAiAgentThought.message_id.in_(message_ids)
        )
        stmt = stmt.order_by(SysAiAgentThought.position.asc())
        result = await db.execute(stmt)
        grouped: dict[int, list[SysAiAgentThought]] = {}
        for t in result.scalars().all():
            grouped.setdefault(t.message_id, []).append(t)
        return grouped

    async def create_thought(
        self,
        db: AsyncSession,
        **kwargs: Any,
    ) -> SysAiAgentThought:
        thought = SysAiAgentThought(**kwargs)
        return await self.create(db, thought)


ai_agent_thought_repository = AiAgentThoughtRepository()
