from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_llm_call import SysAiLlmCall
from app.repository.base import BaseRepository


class AiLlmCallRepository(BaseRepository[SysAiLlmCall]):
    model = SysAiLlmCall

    async def insert_idempotent(self, db: AsyncSession, values: dict) -> None:
        """按 (trace_id, seq) 幂等写入（uk_trace_seq 冲突时忽略，INSERT IGNORE）"""
        stmt = mysql_insert(SysAiLlmCall).values(**values).prefix_with("IGNORE")
        await db.execute(stmt)

    async def list_by_trace(self, db: AsyncSession, trace_id: str) -> list[SysAiLlmCall]:
        """按过程链查询全部 LLM 调用明细（按 seq 正序，回放调用链路）"""
        stmt = (
            select(SysAiLlmCall)
            .where(SysAiLlmCall.trace_id == trace_id)
            .order_by(SysAiLlmCall.seq.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())


ai_llm_call_repository = AiLlmCallRepository()
