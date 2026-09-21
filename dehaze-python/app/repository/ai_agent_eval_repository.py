"""智能体评测数据访问层

覆盖评测集 / 评测样本 / 评测执行记录 / 人工复核四张表的基础查询与写入。
评测集随 Agent 挂载（按 dataset_type 分层），样本属于评测集，运行记录只追加，
复核记录由评测中心按抽样规则生成并回填人工判定。
"""

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.models.entity.sys_ai_agent_eval_dataset import SysAiAgentEvalDataset
from app.models.entity.sys_ai_agent_eval_run import SysAiAgentEvalRun
from app.models.entity.sys_ai_agent_eval_sample import SysAiAgentEvalSample
from app.models.entity.sys_ai_eval_review import SysAiEvalReview
from app.repository.base import BaseRepository

# 评测执行完成状态（1=执行中不参与聚合）
COMPLETED_RUN_STATUSES = (2, 3)


class AiAgentEvalDatasetRepository(BaseRepository[SysAiAgentEvalDataset]):
    model = SysAiAgentEvalDataset

    async def get_by_agent_and_type(
        self,
        db: AsyncSession,
        agent_id: int,
        dataset_type: str,
        *,
        include_deleted: bool = False,
    ) -> SysAiAgentEvalDataset | None:
        """按 (agent_id, dataset_type) 查询评测集（类型内唯一）。"""
        stmt = select(SysAiAgentEvalDataset).where(
            SysAiAgentEvalDataset.agent_id == agent_id,
            SysAiAgentEvalDataset.dataset_type == dataset_type,
        )
        if include_deleted:
            stmt = stmt.execution_options(include_deleted=True)
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

    async def delete_by_datasets(self, db: AsyncSession, dataset_ids: list[int]) -> int:
        """删除评测集下属全部样本（评测集删除时级联清理，样本表无逻辑删除）"""
        if not dataset_ids:
            return 0
        stmt = delete(SysAiAgentEvalSample).where(SysAiAgentEvalSample.dataset_id.in_(dataset_ids))
        result = await db.execute(stmt)
        return result.rowcount


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

    async def delete_by_agent(self, db: AsyncSession, agent_id: int) -> int:
        """删除该 Agent 的全部执行记录（Agent 删除时级联清理，见 agent_service.delete_agent）"""
        stmt = delete(SysAiAgentEvalRun).where(SysAiAgentEvalRun.agent_id == agent_id)
        result = await db.execute(stmt)
        return result.rowcount

    async def get_previous_completed(
        self, db: AsyncSession, agent_id: int, dataset_id: int, exclude_run_id: int
    ) -> SysAiAgentEvalRun | None:
        """该 Agent 同一评测集最近一次已完成评测（相对退化门禁的基准）。"""
        stmt = (
            select(SysAiAgentEvalRun)
            .where(
                SysAiAgentEvalRun.agent_id == agent_id,
                SysAiAgentEvalRun.dataset_id == dataset_id,
                SysAiAgentEvalRun.status.in_(COMPLETED_RUN_STATUSES),
                SysAiAgentEvalRun.id != exclude_run_id,
            )
            .order_by(SysAiAgentEvalRun.id.desc())
            .limit(1)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_latest_per_agent(
        self, db: AsyncSession, per_agent: int = 2
    ) -> list[SysAiAgentEvalRun]:
        """各 Agent 最近 per_agent 次已完成评测（窗口函数，供总览退化判定）。"""
        rn = (
            func.row_number()
            .over(
                partition_by=SysAiAgentEvalRun.agent_id,
                order_by=SysAiAgentEvalRun.id.desc(),
            )
            .label("rn")
        )
        subquery = (
            select(SysAiAgentEvalRun, rn)
            .where(SysAiAgentEvalRun.status.in_(COMPLETED_RUN_STATUSES))
            .subquery()
        )
        run = aliased(SysAiAgentEvalRun, subquery)
        stmt = select(run).where(subquery.c.rn <= per_agent).order_by(run.agent_id, run.id.desc())
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def list_completed(
        self,
        db: AsyncSession,
        agent_id: int | None = None,
        start_time=None,
        end_time=None,
        limit: int = 100,
    ) -> list[SysAiAgentEvalRun]:
        """已完成评测记录（时间升序，供趋势聚合），可按 Agent 与时间范围过滤。"""
        stmt = select(SysAiAgentEvalRun).where(SysAiAgentEvalRun.status.in_(COMPLETED_RUN_STATUSES))
        if agent_id is not None:
            stmt = stmt.where(SysAiAgentEvalRun.agent_id == agent_id)
        if start_time is not None:
            stmt = stmt.where(SysAiAgentEvalRun.create_time >= start_time)
        if end_time is not None:
            stmt = stmt.where(SysAiAgentEvalRun.create_time <= end_time)
        stmt = stmt.order_by(SysAiAgentEvalRun.create_time.asc(), SysAiAgentEvalRun.id.asc()).limit(
            limit
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())


class AiAgentEvalReviewRepository(BaseRepository[SysAiEvalReview]):
    model = SysAiEvalReview

    async def list_by_run_ids(self, db: AsyncSession, run_ids: list[int]) -> list[SysAiEvalReview]:
        if not run_ids:
            return []
        stmt = select(SysAiEvalReview).where(SysAiEvalReview.run_id.in_(run_ids))
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def list_all(self, db: AsyncSession, limit: int = 1000) -> list[SysAiEvalReview]:
        stmt = (
            select(SysAiEvalReview)
            .order_by(SysAiEvalReview.status.asc(), SysAiEvalReview.id.desc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())


ai_agent_eval_dataset_repository = AiAgentEvalDatasetRepository()
ai_agent_eval_sample_repository = AiAgentEvalSampleRepository()
ai_agent_eval_run_repository = AiAgentEvalRunRepository()
ai_agent_eval_review_repository = AiAgentEvalReviewRepository()
