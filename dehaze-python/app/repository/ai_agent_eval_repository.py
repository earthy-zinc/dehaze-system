"""智能体评测数据访问层

覆盖评测集 / 评测样本 / 评测执行记录三张表的基础查询与写入。
评测集随 Agent 挂载（按 dataset_type 分层），样本属于评测集，运行记录只追加。
"""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_agent_eval_dataset import SysAiAgentEvalDataset
from app.models.entity.sys_ai_agent_eval_run import SysAiAgentEvalRun
from app.models.entity.sys_ai_agent_eval_sample import SysAiAgentEvalSample
from app.repository.base import BaseRepository


class AiAgentEvalDatasetRepository(BaseRepository[SysAiAgentEvalDataset]):
    model = SysAiAgentEvalDataset

    async def get_by_agent_and_type(
        self, db: AsyncSession, agent_id: int, dataset_type: str
    ) -> SysAiAgentEvalDataset | None:
        """按 (agent_id, dataset_type) 查询评测集（类型内唯一）。"""
        stmt = select(SysAiAgentEvalDataset).where(
            SysAiAgentEvalDataset.agent_id == agent_id,
            SysAiAgentEvalDataset.dataset_type == dataset_type,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_agent(self, db: AsyncSession, agent_id: int) -> list[SysAiAgentEvalDataset]:
        stmt = (
            select(SysAiAgentEvalDataset)
            .where(SysAiAgentEvalDataset.agent_id == agent_id)
            .order_by(SysAiAgentEvalDataset.id.desc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())


class AiAgentEvalSampleRepository(BaseRepository[SysAiAgentEvalSample]):
    model = SysAiAgentEvalSample

    async def list_by_dataset(
        self, db: AsyncSession, dataset_id: int
    ) -> list[SysAiAgentEvalSample]:
        stmt = (
            select(SysAiAgentEvalSample)
            .where(SysAiAgentEvalSample.dataset_id == dataset_id)
            .order_by(SysAiAgentEvalSample.id.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def count_by_dataset(self, db: AsyncSession, dataset_id: int) -> int:
        stmt = select(func.count()).where(SysAiAgentEvalSample.dataset_id == dataset_id)
        return (await db.execute(stmt)).scalar() or 0


class AiAgentEvalRunRepository(BaseRepository[SysAiAgentEvalRun]):
    model = SysAiAgentEvalRun

    async def list_by_agent(
        self,
        db: AsyncSession,
        agent_id: int,
        page: int,
        size: int,
        dataset_id: int | None = None,
    ) -> tuple[list[SysAiAgentEvalRun], int]:
        stmt = select(SysAiAgentEvalRun).where(SysAiAgentEvalRun.agent_id == agent_id)
        if dataset_id is not None:
            stmt = stmt.where(SysAiAgentEvalRun.dataset_id == dataset_id)
        stmt = stmt.order_by(SysAiAgentEvalRun.id.desc())
        return await self.paginate(db, stmt, page, size)


ai_agent_eval_dataset_repository = AiAgentEvalDatasetRepository()
ai_agent_eval_sample_repository = AiAgentEvalSampleRepository()
ai_agent_eval_run_repository = AiAgentEvalRunRepository()
