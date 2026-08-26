from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_agent_version import SysAiAgentVersion
from app.repository.base import BaseRepository


class AiAgentVersionRepository(BaseRepository[SysAiAgentVersion]):
    model = SysAiAgentVersion

    async def get_latest_published(
        self, db: AsyncSession, agent_id: int
    ) -> SysAiAgentVersion | None:
        stmt = (
            select(SysAiAgentVersion)
            .where(SysAiAgentVersion.agent_id == agent_id, SysAiAgentVersion.status == 2)
            .order_by(SysAiAgentVersion.version_no.desc())
        )
        result = await db.execute(stmt)
        return result.scalars().first()

    async def get_by_agent_and_version(
        self, db: AsyncSession, agent_id: int, version_no: int
    ) -> SysAiAgentVersion | None:
        stmt = select(SysAiAgentVersion).where(
            SysAiAgentVersion.agent_id == agent_id,
            SysAiAgentVersion.version_no == version_no,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def next_version_no(self, db: AsyncSession, agent_id: int) -> int:
        stmt = select(func.max(SysAiAgentVersion.version_no)).where(
            SysAiAgentVersion.agent_id == agent_id
        )
        current = (await db.execute(stmt)).scalar() or 0
        return int(current) + 1

    async def demote_published(self, db: AsyncSession, agent_id: int) -> None:
        """将旧已发布版本置为历史（草稿态，status=0 为历史）。"""
        await db.execute(
            update(SysAiAgentVersion)
            .where(
                SysAiAgentVersion.agent_id == agent_id,
                SysAiAgentVersion.status == 2,
            )
            .values(status=0)
        )

    async def list_versions(self, db: AsyncSession, agent_id: int) -> list[SysAiAgentVersion]:
        stmt = (
            select(SysAiAgentVersion)
            .where(SysAiAgentVersion.agent_id == agent_id)
            .order_by(SysAiAgentVersion.version_no.desc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_published_snapshot(
        self, db: AsyncSession, agent_id: int, version_no: int | None = None
    ) -> SysAiAgentVersion | None:
        """查询已发布版本（未指定版本号取当前已发布版本），供运行面/构建器直接消费。

        返回版本行；调用方经 resolve_snapshot 取得运行面生效配置快照。
        """
        if version_no is None:
            return await self.get_latest_published(db, agent_id)
        return await self.get_by_agent_and_version(db, agent_id, version_no)

    @staticmethod
    def resolve_snapshot(snapshot: dict) -> dict:
        """将快照中冻结的 resolved_config 应用到 config 字段，返回运行面生效配置字典。"""
        resolved = snapshot.get("resolved_config")
        if resolved is None:
            return dict(snapshot)
        result = dict(snapshot)
        result["config"] = resolved
        return result


ai_agent_version_repository = AiAgentVersionRepository()
