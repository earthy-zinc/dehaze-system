"""外部 MCP Server 工具清单数据访问层"""

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_mcp_tool import SysAiMcpTool
from app.repository.base import BaseRepository


class AiMcpToolRepository(BaseRepository[SysAiMcpTool]):
    model = SysAiMcpTool

    async def list_by_server(
        self, db: AsyncSession, server_id: int
    ) -> list[SysAiMcpTool]:
        """查询某 Server 下全部工具，按 id 正序"""
        stmt = (
            select(SysAiMcpTool)
            .where(SysAiMcpTool.server_id == server_id)
            .order_by(SysAiMcpTool.id.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def count_by_server(self, db: AsyncSession, server_id: int) -> int:
        """统计某 Server 下工具数量"""
        stmt = select(SysAiMcpTool.id).where(SysAiMcpTool.server_id == server_id)
        result = await db.execute(stmt)
        return len(result.all())

    async def delete_by_server(self, db: AsyncSession, server_id: int) -> int:
        """级联清理某 Server 下全部工具（重建式覆盖更新用）"""
        stmt = delete(SysAiMcpTool).where(SysAiMcpTool.server_id == server_id)
        result = await db.execute(stmt)
        return result.rowcount


ai_mcp_tool_repository = AiMcpToolRepository()
