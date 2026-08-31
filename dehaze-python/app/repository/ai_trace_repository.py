from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_llm_call import SysAiLlmCall
from app.models.entity.sys_ai_trace import SysAiTrace
from app.repository.base import BaseRepository


class AiTraceRepository(BaseRepository[SysAiTrace]):
    model = SysAiTrace

    async def get_by_trace_id(self, db: AsyncSession, trace_id: str) -> SysAiTrace | None:
        stmt = select(SysAiTrace).where(SysAiTrace.trace_id == trace_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def insert_idempotent(self, db: AsyncSession, values: dict) -> None:
        """按 trace_id 幂等写入（uk_trace_id 冲突时忽略，INSERT IGNORE）"""
        stmt = mysql_insert(SysAiTrace).values(**values).prefix_with("IGNORE")
        await db.execute(stmt)

    async def get_latest_by_message_id(self, db: AsyncSession, message_id: int) -> SysAiTrace | None:
        """查询消息最近一条过程链（resume 续流会产生中断+成功两条，详情取最新）"""
        stmt = (
            select(SysAiTrace)
            .where(SysAiTrace.message_id == message_id)
            .order_by(SysAiTrace.create_time.desc(), SysAiTrace.id.desc())
            .limit(1)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_latest_by_message_and_status(
        self, db: AsyncSession, message_id: int, status: int
    ) -> SysAiTrace | None:
        """查询消息最近一条指定状态的过程链（resume 经 from_trace_id 关联中断过程链）"""
        stmt = (
            select(SysAiTrace)
            .where(SysAiTrace.message_id == message_id, SysAiTrace.status == status)
            .order_by(SysAiTrace.create_time.desc(), SysAiTrace.id.desc())
            .limit(1)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_traces(
        self,
        db: AsyncSession,
        *,
        conversation_id: int | None = None,
        user_id: int | None = None,
        status: int | None = None,
        agent_code: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = 1,
        size: int = 20,
    ) -> tuple[list[SysAiTrace], int]:
        """过程链分页检索（按会话/用户/状态/智能体/时间筛选，用户维度经会话表关联）"""
        stmt = select(SysAiTrace)
        if conversation_id is not None:
            stmt = stmt.where(SysAiTrace.conversation_id == conversation_id)
        if user_id is not None:
            stmt = stmt.join(
                SysAiConversation, SysAiTrace.conversation_id == SysAiConversation.id
            ).where(SysAiConversation.user_id == user_id, SysAiConversation.deleted == 0)
        if status is not None:
            stmt = stmt.where(SysAiTrace.status == status)
        if agent_code is not None:
            stmt = stmt.where(SysAiTrace.agent_code == agent_code)
        if start_time is not None:
            stmt = stmt.where(SysAiTrace.create_time >= start_time)
        if end_time is not None:
            stmt = stmt.where(SysAiTrace.create_time <= end_time)
        stmt = stmt.order_by(SysAiTrace.create_time.desc(), SysAiTrace.id.desc())
        return await self.paginate(db, stmt, page, size)

    async def list_abnormal_conversation_ids(
        self, db: AsyncSession, conv_ids: list[int]
    ) -> set[int]:
        """查询存在失败/中断/超时过程链的会话 ID 集合（会话审计页异常标注）"""
        if not conv_ids:
            return set()
        stmt = (
            select(SysAiTrace.conversation_id)
            .where(
                SysAiTrace.conversation_id.in_(conv_ids),
                SysAiTrace.status.in_((2, 3, 4)),
            )
            .group_by(SysAiTrace.conversation_id)
        )
        result = await db.execute(stmt)
        return {row[0] for row in result.all()}

    async def list_risky_tool_conversation_ids(
        self, db: AsyncSession, conv_ids: list[int]
    ) -> set[int]:
        """存在高风险工具调用的会话 ID 集合（会话审计异常标注的 risky_tool 数据源）。

        口径：过程链存在"发起工具调用但调用失败/超时"的 LLM 调用
        （sys_ai_llm_call.tool_call 非空且 status != 1，工具失败不落 error_type）。
        """
        if not conv_ids:
            return set()
        stmt = (
            select(SysAiTrace.conversation_id)
            .join(SysAiLlmCall, SysAiLlmCall.trace_id == SysAiTrace.trace_id)
            .where(
                SysAiTrace.conversation_id.in_(conv_ids),
                SysAiLlmCall.tool_call.is_not(None),
                SysAiLlmCall.status != 1,
            )
            .distinct()
        )
        result = await db.execute(stmt)
        return {row[0] for row in result.all()}

    async def count_by_status(self, db: AsyncSession) -> dict[int, int]:
        """按状态统计过程链数量（聚合查询辅助）"""
        stmt = select(SysAiTrace.status, func.count()).group_by(SysAiTrace.status)
        result = await db.execute(stmt)
        return {row[0]: row[1] for row in result.all()}


ai_trace_repository = AiTraceRepository()
