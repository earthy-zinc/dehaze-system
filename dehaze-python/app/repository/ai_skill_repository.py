"""AI 对话 Skill 主表仓库（F-M08-006 Skills 管理）。

提供 Skill 的查询（按名/按状态/分页）、写入、软删，以及 Agent 关联校验
（count_by_names 供 Agent set_skills 存在性校验、count_agent_references 供删除前校验）。
逻辑删除字段由全局 do_orm_execute 事件自动过滤（继承 SoftDeleteMixin）。
"""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_agent_skill import SysAiAgentSkill
from app.models.entity.sys_ai_skill import SysAiSkill
from app.repository.base import BaseRepository, escape_like


class AiSkillRepository(BaseRepository[SysAiSkill]):
    model = SysAiSkill

    async def get_by_name(self, db: AsyncSession, name: str) -> SysAiSkill | None:
        """按名称查询（不含已删，用于唯一性校验与获取）。"""
        stmt = select(SysAiSkill).where(SysAiSkill.name == name)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_name_with_deleted(self, db: AsyncSession, name: str) -> SysAiSkill | None:
        """按名称查询（含已删，删除后 name 不可复用，用于判重）。"""
        stmt = select(SysAiSkill).where(SysAiSkill.name == name)
        stmt = stmt.execution_options(include_deleted=True)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(self, db: AsyncSession, status: int | None = None) -> list[SysAiSkill]:
        """按状态过滤查询全部（status=None 返回全部未删项）。"""
        stmt = select(SysAiSkill).order_by(SysAiSkill.id.asc())
        if status is not None:
            stmt = stmt.where(SysAiSkill.status == status)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def page(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        keyword: str | None = None,
    ) -> tuple[list[SysAiSkill], int]:
        """分页 + 名称模糊搜索（含全部状态，按 id 倒序）。"""
        stmt = select(SysAiSkill)
        if keyword:
            escaped = escape_like(keyword)
            stmt = stmt.where(SysAiSkill.name.like(f"%{escaped}%", escape="\\"))
        stmt = stmt.order_by(SysAiSkill.id.desc())
        return await self.paginate(db, stmt, page, size)

    async def count_by_names(self, db: AsyncSession, names: list[str]) -> int:
        """统计给定名称中存在于主表（未删）的 Skill 数量（Agent 关联存在性校验用）。"""
        if not names:
            return 0
        stmt = select(func.count()).select_from(SysAiSkill).where(SysAiSkill.name.in_(names))
        return (await db.execute(stmt)).scalar() or 0

    async def list_names_existing(self, db: AsyncSession, names: list[str]) -> list[str]:
        """返回给定名称中存在于主表（未删）的 Skill 名列表（供缺失项精确提示）。"""
        if not names:
            return []
        stmt = select(SysAiSkill.name).where(SysAiSkill.name.in_(names))
        rows = (await db.execute(stmt)).scalars().all()
        return list(rows)

    async def count_agent_references(self, db: AsyncSession, skill_name: str) -> int:
        """统计被 Agent 关联的条数（删除前校验，sys_ai_agent_skill.skill_name）。"""
        stmt = (
            select(func.count())
            .select_from(SysAiAgentSkill)
            .where(SysAiAgentSkill.skill_name == skill_name)
        )
        return (await db.execute(stmt)).scalar() or 0


ai_skill_repository = AiSkillRepository()
