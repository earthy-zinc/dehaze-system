"""外部 A2A 端点注册数据访问层"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_agent_endpoint import SysAiAgentEndpoint
from app.repository.base import BaseRepository, escape_like


class AiAgentEndpointRepository(BaseRepository[SysAiAgentEndpoint]):
    model = SysAiAgentEndpoint

    async def get_by_base_url(self, db: AsyncSession, base_url: str) -> SysAiAgentEndpoint | None:
        stmt = select(SysAiAgentEndpoint).where(SysAiAgentEndpoint.base_url == base_url)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def paginate_endpoints(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        keyword: str | None = None,
        status: int | None = None,
    ) -> tuple[list[SysAiAgentEndpoint], int]:
        stmt = select(SysAiAgentEndpoint)
        if keyword:
            escaped = escape_like(keyword)
            pattern = f"%{escaped}%"
            stmt = stmt.where(
                (SysAiAgentEndpoint.name.like(pattern, escape="\\"))
                | (SysAiAgentEndpoint.base_url.like(pattern, escape="\\"))
            )
        if status is not None:
            stmt = stmt.where(SysAiAgentEndpoint.status == status)
        stmt = stmt.order_by(SysAiAgentEndpoint.id.desc())
        return await self.paginate(db, stmt, page, size)

    async def list_enabled(self, db: AsyncSession) -> list[SysAiAgentEndpoint]:
        stmt = (
            select(SysAiAgentEndpoint)
            .where(SysAiAgentEndpoint.status == 1)
            .order_by(SysAiAgentEndpoint.id.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())


ai_agent_endpoint_repository = AiAgentEndpointRepository()
