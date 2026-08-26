"""外部 MCP 工具调用审计数据访问层"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_mcp_call import SysAiMcpCall
from app.repository.base import BaseRepository, escape_like


class AiMcpCallRepository(BaseRepository[SysAiMcpCall]):
    model = SysAiMcpCall

    async def paginate_calls(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        server_id: int | None = None,
        tool_name: str | None = None,
        user_id: int | None = None,
    ) -> tuple[list[SysAiMcpCall], int]:
        """分页查询调用审计（create_time 倒序；server_id/tool_name/user_id 筛选）"""
        stmt = select(SysAiMcpCall)
        if server_id is not None:
            stmt = stmt.where(SysAiMcpCall.server_id == server_id)
        if tool_name:
            escaped = escape_like(tool_name)
            stmt = stmt.where(SysAiMcpCall.tool_name.like(f"%{escaped}%", escape="\\"))
        if user_id is not None:
            stmt = stmt.where(SysAiMcpCall.user_id == user_id)
        stmt = stmt.order_by(SysAiMcpCall.id.desc())
        return await self.paginate(db, stmt, page, size)


ai_mcp_call_repository = AiMcpCallRepository()
