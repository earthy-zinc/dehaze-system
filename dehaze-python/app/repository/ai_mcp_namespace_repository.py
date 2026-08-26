"""外部 MCP Server 命名空间数据访问层"""

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_mcp_namespace import SysAiMcpNamespace
from app.repository.base import BaseRepository


class AiMcpNamespaceRepository(BaseRepository[SysAiMcpNamespace]):
    model = SysAiMcpNamespace

    async def list_by_server(
        self, db: AsyncSession, server_id: int
    ) -> list[SysAiMcpNamespace]:
        """查询某 Server 下全部命名空间，按 id 正序"""
        stmt = (
            select(SysAiMcpNamespace)
            .where(SysAiMcpNamespace.server_id == server_id)
            .order_by(SysAiMcpNamespace.id.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def delete_by_server(self, db: AsyncSession, server_id: int) -> int:
        """级联清理某 Server 下全部命名空间（覆盖式更新/删除时用）"""
        stmt = delete(SysAiMcpNamespace).where(SysAiMcpNamespace.server_id == server_id)
        result = await db.execute(stmt)
        return result.rowcount


ai_mcp_namespace_repository = AiMcpNamespaceRepository()
