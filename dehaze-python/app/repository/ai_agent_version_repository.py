from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import load_only

from app.models.entity.sys_ai_agent_version import SysAiAgentVersion
from app.repository.base import BaseRepository

# 版本列表不加载 snapshot（快照为完整配置 JSON，列表接口逐行反序列化代价高）
_LIST_COLUMNS = (
    SysAiAgentVersion.id,
    SysAiAgentVersion.agent_id,
    SysAiAgentVersion.version_no,
    SysAiAgentVersion.status,
    SysAiAgentVersion.change_note,
    SysAiAgentVersion.operator_id,
    SysAiAgentVersion.create_time,
)


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

    async def get_latest_draft(self, db: AsyncSession, agent_id: int) -> SysAiAgentVersion | None:
        """查询最新草稿版本（评测门禁对象：即将生效的草稿配置，见评测后端实现 §1.1）。"""
        stmt = (
            select(SysAiAgentVersion)
            .where(SysAiAgentVersion.agent_id == agent_id, SysAiAgentVersion.status == 1)
            .order_by(SysAiAgentVersion.version_no.desc())
            .limit(1)
        )
        return (await db.execute(stmt)).scalars().first()

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

    async def list_versions(
        self, db: AsyncSession, agent_id: int, offset: int, limit: int
    ) -> tuple[list[SysAiAgentVersion], int]:
        """分页查询版本历史，返回 (当前页, 总数)。

        分页下推到 SQL：版本行随发布持续增长，全量加载后在内存切片会把整个版本历史的
        快照 JSON 都读进内存。
        """
        where = SysAiAgentVersion.agent_id == agent_id
        total = (
            await db.execute(select(func.count(SysAiAgentVersion.id)).where(where))
        ).scalar() or 0
        if not total:
            return [], 0
        stmt = (
            select(SysAiAgentVersion)
            .where(where)
            .options(load_only(*_LIST_COLUMNS))
            .order_by(SysAiAgentVersion.version_no.desc())
            .offset(offset)
            .limit(limit)
        )
        return list((await db.execute(stmt)).scalars().all()), int(total)

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
