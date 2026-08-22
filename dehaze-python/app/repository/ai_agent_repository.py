from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_agent import SysAiAgent
from app.models.entity.sys_ai_agent_mcp import SysAiAgentMcp
from app.models.entity.sys_ai_agent_skill import SysAiAgentSkill
from app.models.entity.sys_ai_agent_subagent import SysAiAgentSubagent
from app.repository.base import BaseRepository, escape_like


class AiAgentRepository(BaseRepository[SysAiAgent]):
    model = SysAiAgent

    async def get_by_code(
        self,
        db: AsyncSession,
        agent_code: str,
        *,
        with_deleted: bool = False,
    ) -> SysAiAgent | None:
        """按业务唯一键 agent_code 查询（类别②：绕过软删查全表，删除后不可复用）。"""
        stmt = select(SysAiAgent).where(SysAiAgent.agent_code == agent_code)
        if with_deleted:
            stmt = stmt.execution_options(include_deleted=True)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id_with_deleted(
        self,
        db: AsyncSession,
        agent_id: int,
    ) -> SysAiAgent | None:
        """按主键查询（含已删除，用于删除态判断）。"""
        stmt = select(SysAiAgent).where(SysAiAgent.id == agent_id)
        stmt = stmt.execution_options(include_deleted=True)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def paginate_agents(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        keyword: str | None = None,
        status: int | None = None,
    ) -> tuple[list[SysAiAgent], int]:
        stmt = select(SysAiAgent)
        if keyword:
            escaped = escape_like(keyword)
            pattern = f"%{escaped}%"
            stmt = stmt.where(
                (SysAiAgent.name.like(pattern, escape="\\"))
                | (SysAiAgent.agent_code.like(pattern, escape="\\"))
            )
        if status is not None:
            stmt = stmt.where(SysAiAgent.status == status)
        stmt = stmt.order_by(SysAiAgent.sort_order.asc(), SysAiAgent.id.asc())
        return await self.paginate(db, stmt, page, size)

    async def list_enabled(self, db: AsyncSession) -> list[SysAiAgent]:
        """可选 Agent 列表：启用且非子 Agent（Team 可作会话入口，保留）。"""
        stmt = (
            select(SysAiAgent)
            .where(SysAiAgent.status == 1, SysAiAgent.is_subagent == 0)
            .order_by(SysAiAgent.sort_order.asc(), SysAiAgent.id.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def count_subagent_references(self, db: AsyncSession, agent_id: int) -> int:
        """统计该 Agent 被其他 Agent 作为子 Agent 引用的次数（删除校验）。"""
        stmt = (
            select(func.count())
            .select_from(SysAiAgentSubagent)
            .where(SysAiAgentSubagent.subagent_agent_id == agent_id)
        )
        return (await db.execute(stmt)).scalar() or 0

    async def count_conversation_references(self, db: AsyncSession, agent_code: str) -> int:
        """统计使用该 Agent 的活跃会话数（删除校验，按会话锚定的 agent_code 查询）。"""
        from app.models.entity.sys_ai_conversation import SysAiConversation

        stmt = (
            select(func.count())
            .select_from(SysAiConversation)
            .where(
                SysAiConversation.agent_code == agent_code,
                SysAiConversation.deleted == 0,
            )
        )
        return (await db.execute(stmt)).scalar() or 0

    # ── Skills / MCP / 子 Agent 关联（覆盖式更新）────────────────

    async def replace_skills(self, db: AsyncSession, agent_id: int, skill_names: list[str]) -> None:
        await db.execute(delete(SysAiAgentSkill).where(SysAiAgentSkill.agent_id == agent_id))
        for skill_name in skill_names:
            db.add(SysAiAgentSkill(agent_id=agent_id, skill_name=skill_name))

    async def list_skill_names(self, db: AsyncSession, agent_id: int) -> list[str]:
        stmt = (
            select(SysAiAgentSkill.skill_name)
            .where(SysAiAgentSkill.agent_id == agent_id)
            .order_by(SysAiAgentSkill.skill_name)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def replace_mcp_namespaces(
        self, db: AsyncSession, agent_id: int, mcp_namespaces: list[str]
    ) -> None:
        await db.execute(delete(SysAiAgentMcp).where(SysAiAgentMcp.agent_id == agent_id))
        for ns in mcp_namespaces:
            db.add(SysAiAgentMcp(agent_id=agent_id, mcp_namespace=ns))

    async def list_mcp_namespaces(self, db: AsyncSession, agent_id: int) -> list[str]:
        stmt = (
            select(SysAiAgentMcp.mcp_namespace)
            .where(SysAiAgentMcp.agent_id == agent_id)
            .order_by(SysAiAgentMcp.mcp_namespace)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def replace_subagents(self, db: AsyncSession, agent_id: int, items: list[dict]) -> None:
        """覆盖式更新子 Agent 关联（items: {agent_id, endpoint_id, priority}）。"""
        await db.execute(
            delete(SysAiAgentSubagent).where(SysAiAgentSubagent.parent_agent_id == agent_id)
        )
        for item in items:
            db.add(
                SysAiAgentSubagent(
                    parent_agent_id=agent_id,
                    subagent_agent_id=item["agent_id"],
                    endpoint_id=item.get("endpoint_id"),
                    priority=item.get("priority", 0),
                )
            )

    async def list_subagents(
        self, db: AsyncSession, parent_agent_id: int
    ) -> list[SysAiAgentSubagent]:
        stmt = (
            select(SysAiAgentSubagent)
            .where(SysAiAgentSubagent.parent_agent_id == parent_agent_id)
            .order_by(SysAiAgentSubagent.priority.asc(), SysAiAgentSubagent.create_time)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())


ai_agent_repository = AiAgentRepository()
