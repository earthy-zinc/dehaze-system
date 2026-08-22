"""智能体评测服务（EvalService）

评测集 / 样本 CRUD + 评测执行 + 发布门禁判定。

评测执行流程（run_regression）：
    建 eval_run(status=1) → 取已发布快照 → 逐样本执行（EvalRunner）→
    四维评分汇总 score_summary + results → 更新 eval_run(status=2/3) →
    返回门禁判定 {"passed", "score_summary", "failed_samples"}。

门禁规则：任一维度低于阈值，或 risk_level=high 样本失败 → passed=False。
评测不计入用户配额（独立会话上下文 + 平台专用 Token 池，计费隔离）。
"""

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.base import get_current_user_id
from app.models.entity.sys_ai_agent_eval_dataset import SysAiAgentEvalDataset
from app.models.entity.sys_ai_agent_eval_run import SysAiAgentEvalRun
from app.models.entity.sys_ai_agent_eval_sample import SysAiAgentEvalSample
from app.repository.ai_agent_eval_repository import (
    ai_agent_eval_dataset_repository,
    ai_agent_eval_run_repository,
    ai_agent_eval_sample_repository,
)
from app.service.ai.eval_runner import eval_runner
from app.service.ai_agent_service import AgentService

logger = logging.getLogger(__name__)


class EvalService:
    @staticmethod
    async def _get_agent_or_raise(db: AsyncSession, agent_id: int) -> None:
        from app.repository.ai_agent_repository import ai_agent_repository

        agent = await ai_agent_repository.get_by_id(db, agent_id)
        if not agent:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "Agent 不存在")

    # ── 评测集 CRUD ────────────────────────────────────────────

    @staticmethod
    async def create_dataset(db: AsyncSession, agent_id: int, form) -> SysAiAgentEvalDataset:
        await EvalService._get_agent_or_raise(db, agent_id)
        if await ai_agent_eval_dataset_repository.get_by_agent_and_type(
            db, agent_id, form.dataset_type
        ):
            raise BusinessException(ResultCode.DATA_EXISTS, "该 Agent 已存在同类型评测集")
        dataset = SysAiAgentEvalDataset(
            agent_id=agent_id,
            name=form.name,
            description=form.description,
            dataset_type=form.dataset_type,
        )
        return await ai_agent_eval_dataset_repository.create(db, dataset)

    @staticmethod
    async def update_dataset(db: AsyncSession, dataset_id: int, form) -> SysAiAgentEvalDataset:
        dataset = await ai_agent_eval_dataset_repository.get_by_id(db, dataset_id)
        if not dataset:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "评测集不存在")
        data = form.model_dump(exclude_unset=True)
        for field, value in data.items():
            setattr(dataset, field, value)
        await db.flush()
        await db.refresh(dataset)
        return dataset

    @staticmethod
    async def delete_dataset(db: AsyncSession, dataset_id: int) -> None:
        dataset = await ai_agent_eval_dataset_repository.get_by_id(db, dataset_id)
        if not dataset:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "评测集不存在")
        await ai_agent_eval_dataset_repository.soft_delete_by_ids(db, [dataset_id])

    @staticmethod
    async def list_datasets(db: AsyncSession, agent_id: int) -> list[SysAiAgentEvalDataset]:
        return await ai_agent_eval_dataset_repository.list_by_agent(db, agent_id)

    # ── 样本 CRUD ──────────────────────────────────────────────

    @staticmethod
    async def _get_dataset_or_raise(db: AsyncSession, dataset_id: int) -> SysAiAgentEvalDataset:
        dataset = await ai_agent_eval_dataset_repository.get_by_id(db, dataset_id)
        if not dataset:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "评测集不存在")
        return dataset

    @staticmethod
    async def create_sample(db: AsyncSession, dataset_id: int, form) -> SysAiAgentEvalSample:
        await EvalService._get_dataset_or_raise(db, dataset_id)
        sample = SysAiAgentEvalSample(
            dataset_id=dataset_id,
            task_goal=form.task_goal,
            allowed_input=form.allowed_input,
            tools=form.tools,
            expected_process=form.expected_process,
            expected_result=form.expected_result,
            forbidden_behavior=form.forbidden_behavior,
            risk_level=form.risk_level,
        )
        return await ai_agent_eval_sample_repository.create(db, sample)

    @staticmethod
    async def update_sample(db: AsyncSession, sample_id: int, form) -> SysAiAgentEvalSample:
        sample = await ai_agent_eval_sample_repository.get_by_id(db, sample_id)
        if not sample:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "评测样本不存在")
        data = form.model_dump(exclude_unset=True)
        for field, value in data.items():
            setattr(sample, field, value)
        await db.flush()
        await db.refresh(sample)
        return sample

    @staticmethod
    async def delete_sample(db: AsyncSession, sample_id: int) -> None:
        sample = await ai_agent_eval_sample_repository.get_by_id(db, sample_id)
        if not sample:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "评测样本不存在")
        await ai_agent_eval_sample_repository.delete_by_ids(db, [sample_id])

    @staticmethod
    async def list_samples(db: AsyncSession, dataset_id: int) -> list[SysAiAgentEvalSample]:
        await EvalService._get_dataset_or_raise(db, dataset_id)
        return await ai_agent_eval_sample_repository.list_by_dataset(db, dataset_id)

    # ── 评测执行 ───────────────────────────────────────────────

    @staticmethod
    async def run_regression(
        db: AsyncSession,
        redis,
        agent_id: int,
        trigger_type: str = "publish",
    ) -> dict:
        """跑回归评测集并返回门禁判定。

        契约（供 AgentVersionService.publish 调用）：回归集缺失或样本为空时视为通过
        （无考题则门禁放行），有样本则执行并汇总。
        """
        await EvalService._get_agent_or_raise(db, agent_id)
        dataset = await ai_agent_eval_dataset_repository.get_by_agent_and_type(
            db, agent_id, "regression"
        )
        # 新 Agent 首发无回归集时平凡放行，避免严格阻断造成发布死锁；
        # 后续配置了回归集后，发布门禁按实际评测结果判定。
        if not dataset:
            return {"passed": True, "score_summary": {}, "failed_samples": [], "run_id": None}

        samples = await ai_agent_eval_sample_repository.list_by_dataset(db, dataset.id)
        # 评测集为空（未录入样本）同样视为无考题，门禁放行。
        if not samples:
            return {"passed": True, "score_summary": {}, "failed_samples": [], "run_id": None}

        # 已发布快照：整批样本基于同一配置评测
        snapshot = await AgentService().get_published_snapshot(db, redis, agent_id, None)

        run = SysAiAgentEvalRun(
            agent_id=agent_id,
            dataset_id=dataset.id,
            trigger_type=trigger_type,
            status=1,
            create_by=get_current_user_id(),
        )
        run = await ai_agent_eval_run_repository.create(db, run)

        results = []
        for sample in samples:
            try:
                result = await eval_runner.run_sample(db, redis, sample, snapshot)
            except Exception as exc:  # noqa: BLE001 - 单样本失败计为该样本失败
                logger.warning("评测样本 %s 执行异常: %s", sample.id, exc, exc_info=True)
                result = {
                    "sample_id": sample.id,
                    "task_goal": sample.task_goal,
                    "risk_level": sample.risk_level,
                    "passed": False,
                    "error": str(exc),
                    "scores": {
                        "result_quality": 0,
                        "process_compliance": 0,
                        "safety_boundary": 0,
                        "efficiency": 0,
                    },
                    "notes": {},
                    "metrics": {},
                }
            results.append(result)

        score_summary = _aggregate_scores(results)
        failed_samples = [r for r in results if not r["passed"]]
        # 门禁：任一维度低于阈值或 high 风险样本失败（EvalRunner 已判定），有失败样本即阻断
        passed = not failed_samples
        run.status = 2 if passed else 3
        run.score_summary = score_summary
        run.results = results
        await db.flush()
        await db.refresh(run)

        return {
            "passed": passed,
            "score_summary": score_summary,
            "failed_samples": failed_samples,
            "run_id": run.id,
        }

    @staticmethod
    async def list_runs(
        db: AsyncSession,
        agent_id: int,
        page: int,
        size: int,
        dataset_id: int | None = None,
    ) -> tuple[list[SysAiAgentEvalRun], int]:
        return await ai_agent_eval_run_repository.list_by_agent(
            db, agent_id, page, size, dataset_id
        )


def _aggregate_scores(results: list[dict]) -> dict[str, Any]:
    """聚合四维评分为均值，并统计通过率。"""
    if not results:
        return {}
    dimensions = ("result_quality", "process_compliance", "safety_boundary", "efficiency")
    aggregated = {
        dim: round(sum(r["scores"].get(dim, 0) for r in results) / len(results), 2)
        for dim in dimensions
    }
    passed_count = sum(1 for r in results if r["passed"])
    return {
        "dimensions": aggregated,
        "sample_count": len(results),
        "passed_count": passed_count,
        "failed_count": len(results) - passed_count,
        "pass_rate": round(passed_count / len(results), 4),
    }


eval_service = EvalService()
