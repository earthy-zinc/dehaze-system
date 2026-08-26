"""外部 MCP Server 注册数据访问层"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_mcp_server import SysAiMcpServer
from app.repository.base import BaseRepository, escape_like


class AiMcpServerRepository(BaseRepository[SysAiMcpServer]):
    model = SysAiMcpServer

    async def get_by_name(
        self, db: AsyncSession, name: str, include_deleted: bool = False
    ) -> SysAiMcpServer | None:
        """按业务唯一键 name 查询（查重时 include_deleted=True 绕过软删过滤）"""
        stmt = select(SysAiMcpServer).where(SysAiMcpServer.name == name)
        if include_deleted:
            stmt = stmt.execution_options(include_deleted=True)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def paginate_servers(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        keyword: str | None = None,
        status: int | None = None,
    ) -> tuple[list[SysAiMcpServer], int]:
        """分页查询 Server（keyword 匹配 name/description）"""
        stmt = select(SysAiMcpServer)
        if keyword:
            escaped = escape_like(keyword)
            pattern = f"%{escaped}%"
            stmt = stmt.where(
                (SysAiMcpServer.name.like(pattern, escape="\\"))
                | (SysAiMcpServer.description.like(pattern, escape="\\"))
            )
        if status is not None:
            stmt = stmt.where(SysAiMcpServer.status == status)
        stmt = stmt.order_by(SysAiMcpServer.id.desc())
        return await self.paginate(db, stmt, page, size)

    async def list_enabled(self, db: AsyncSession) -> list[SysAiMcpServer]:
        """查询启用的 Server（status=1 且未删除），按 id 正序"""
        stmt = select(SysAiMcpServer).where(SysAiMcpServer.status == 1)
        result = await db.execute(stmt)
        return list(result.scalars().all())


ai_mcp_server_repository = AiMcpServerRepository()
