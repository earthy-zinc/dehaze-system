"""智能体评测服务（EvalService）

评测集 / 样本 CRUD + 评测执行 + 发布门禁判定。

评测执行流程（_execute_regression）：
    建 eval_run(status=1) → 取草稿快照 → 逐样本执行（EvalRunner）→
    四维评分汇总 score_summary + results → 更新 eval_run(status=2/3) →
    返回门禁判定 {"passed", "score_summary", "failed_samples"}。

门禁规则：任一维度低于阈值，或 risk_level=high 样本失败 → passed=False。

执行方式：手动触发走异步任务（POST /runs 立即返回 task_id，进度与结果查
GET /tasks/{task_id}），发布门禁走同步 run_regression（发布链路需要即时结论）。
评测不计入用户配额（独立会话上下文 + 平台专用 Token 池，计费隔离）。
"""

import asyncio
import json
import logging
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db_session
from app.models.base import get_current_user_id
from app.models.entity.sys_ai_agent_eval_dataset import SysAiAgentEvalDataset
from app.models.entity.sys_ai_agent_eval_run import SysAiAgentEvalRun
from app.models.entity.sys_ai_agent_eval_sample import SysAiAgentEvalSample
from app.repository.ai_agent_eval_repository import (
    ai_agent_eval_dataset_repository,
    ai_agent_eval_run_repository,
    ai_agent_eval_sample_repository,
)
from app.repository.ai_agent_version_repository import ai_agent_version_repository
from app.repository.mongo_audit_log_repository import mongo_audit_log_repository
from app.service.ai.service.eval_runner import JUDGE_MODEL, eval_runner
from app.service.ai_eval_center_service import (
    DICT_TYPE_AI_EVAL,
    REGRESSION_THRESHOLD_DEFAULT,
    _is_degraded,
    _total_score,
)
from app.service.dict_service import get_dict_int

logger = logging.getLogger(__name__)

# 异步评测任务状态键与保留期（Redis 而非进程内存：进度查询可能落到任意实例）
EVAL_TASK_KEY_PREFIX = "ai:eval:task:"
EVAL_TASK_TTL = 24 * 3600

# 门禁样本数下限：回归集已配置却无考题时阻断发布，不静默放行
EVAL_GATE_MIN_SAMPLES = 1

# 在途评测任务强引用（事件循环仅弱引用 task，不持有会被 GC 提前取消）
_EVAL_TASKS: set[asyncio.Task] = set()


def _task_key(task_id: str) -> str:
    return EVAL_TASK_KEY_PREFIX + task_id


async def _update_task(redis, task_id: str, **fields: Any) -> None:
    """读改写任务状态：progress 由逐样本推进，未覆盖字段保持原值。"""
    raw = await redis.get(_task_key(task_id))
    payload = json.loads(raw) if raw else {}
    payload.update(fields)
    await redis.set(_task_key(task_id), json.dumps(payload, ensure_ascii=False), ex=EVAL_TASK_TTL)


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
        existing = await ai_agent_eval_dataset_repository.get_by_agent_and_type(
            db, agent_id, form.dataset_type, include_deleted=True
        )
        if existing:
            if existing.deleted == 0:
                raise BusinessException(ResultCode.DATA_EXISTS, "该 Agent 已存在同类型评测集")
            # 软删行占用唯一键 (agent_id, dataset_type)，复活原行而非插入
            existing.name = form.name
            existing.description = form.description
            existing.deleted = 0
            await db.flush()
            await db.refresh(existing)
            return existing
        dataset = SysAiAgentEvalDataset(
            agent_id=agent_id,
            name=form.name,
            description=form.description,
            dataset_type=form.dataset_type,
        )
        return await ai_agent_eval_dataset_repository.create(db, dataset)

    @staticmethod
    async def update_dataset(
        db: AsyncSession, agent_id: int, dataset_id: int, form
    ) -> SysAiAgentEvalDataset:
        dataset = await EvalService._get_dataset_of_agent_or_raise(db, agent_id, dataset_id)
        data = form.model_dump(exclude_unset=True)
        for field, value in data.items():
            setattr(dataset, field, value)
        await db.flush()
        await db.refresh(dataset)
        return dataset

    @staticmethod
    async def delete_dataset(
        db: AsyncSession, agent_id: int, dataset_id: int, operator_id: int
    ) -> None:
        dataset = await EvalService._get_dataset_of_agent_or_raise(db, agent_id, dataset_id)
        # 样本随数据集管理：数据集软删除的同时级联清理样本（样本表无逻辑删除）
        await ai_agent_eval_sample_repository.delete_by_datasets(db, [dataset.id])
        await ai_agent_eval_dataset_repository.soft_delete_by_ids(db, [dataset.id])
        mongo_audit_log_repository.create_audit_async(
            operator_id=operator_id,
            target_type="ai_eval_dataset",
            target_id=dataset.id,
            action="delete",
            module="ai_eval",
            before_value={
                "agent_id": agent_id,
                "name": dataset.name,
                "dataset_type": dataset.dataset_type,
            },
        )

    @staticmethod
    async def list_datasets(db: AsyncSession, agent_id: int) -> list[SysAiAgentEvalDataset]:
        return await ai_agent_eval_dataset_repository.list_by_agent(db, agent_id)

    # ── 样本 CRUD ──────────────────────────────────────────────

    @staticmethod
    async def _get_dataset_of_agent_or_raise(
        db: AsyncSession, agent_id: int, dataset_id: int
    ) -> SysAiAgentEvalDataset:
        """评测集必须归属于路径中的 Agent（跨 Agent 操作一律 404，不暴露存在性）。"""
        dataset = await ai_agent_eval_dataset_repository.get_by_id(db, dataset_id)
        if not dataset or dataset.agent_id != agent_id:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "评测集不存在")
        return dataset

    @staticmethod
    async def _get_sample_of_agent_or_raise(
        db: AsyncSession, agent_id: int, sample_id: int
    ) -> SysAiAgentEvalSample:
        sample = await ai_agent_eval_sample_repository.get_by_id(db, sample_id)
        if not sample:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "评测样本不存在")
        await EvalService._get_dataset_of_agent_or_raise(db, agent_id, sample.dataset_id)
        return sample

    @staticmethod
    async def create_sample(
        db: AsyncSession, agent_id: int, dataset_id: int, form
    ) -> SysAiAgentEvalSample:
        if form.dataset_id != dataset_id:
            raise BusinessException(ResultCode.PARAM_ERROR, "样本所属评测集与路径不一致")
        await EvalService._get_dataset_of_agent_or_raise(db, agent_id, dataset_id)
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
    async def update_sample(
        db: AsyncSession, agent_id: int, sample_id: int, form
    ) -> SysAiAgentEvalSample:
        sample = await EvalService._get_sample_of_agent_or_raise(db, agent_id, sample_id)
        data = form.model_dump(exclude_unset=True)
        for field, value in data.items():
            setattr(sample, field, value)
        await db.flush()
        await db.refresh(sample)
        return sample

    @staticmethod
    async def delete_sample(
        db: AsyncSession, agent_id: int, sample_id: int, operator_id: int
    ) -> None:
        sample = await EvalService._get_sample_of_agent_or_raise(db, agent_id, sample_id)
        await ai_agent_eval_sample_repository.delete_by_ids(db, [sample.id])
        mongo_audit_log_repository.create_audit_async(
            operator_id=operator_id,
            target_type="ai_eval_sample",
            target_id=sample.id,
            action="delete",
            module="ai_eval",
            before_value={"dataset_id": sample.dataset_id, "task_goal": sample.task_goal},
        )

    @staticmethod
    async def list_samples(
        db: AsyncSession, agent_id: int, dataset_id: int
    ) -> list[SysAiAgentEvalSample]:
        await EvalService._get_dataset_of_agent_or_raise(db, agent_id, dataset_id)
        return await ai_agent_eval_sample_repository.list_by_dataset(db, dataset_id)

    # ── 评测执行 ───────────────────────────────────────────────

    @staticmethod
    async def start_eval_task(
        db: AsyncSession,
        redis,
        agent_id: int,
        operator_id: int,
        trigger_type: str = "manual",
    ) -> str:
        """登记异步评测任务并立即返回 task_id（评测为分钟级长跑，不占用请求连接）。

        任务状态落 Redis（ai:eval:task:{task_id}），进度与结果由 GET /tasks/{task_id} 查询。
        """
        await EvalService._get_agent_or_raise(db, agent_id)
        task_id = uuid.uuid4().hex
        await _update_task(
            redis,
            task_id,
            status="pending",
            progress={"done": 0, "total": 0},
            result=None,
            error=None,
        )
        mongo_audit_log_repository.create_audit_async(
            operator_id=operator_id,
            target_type="ai_eval_run",
            target_id=task_id,
            action="start",
            module="ai_eval",
            after_value={"agent_id": agent_id, "trigger_type": trigger_type},
        )
        task = asyncio.create_task(
            EvalService._execute_task(task_id, redis, agent_id, trigger_type, operator_id)
        )
        _EVAL_TASKS.add(task)
        task.add_done_callback(_EVAL_TASKS.discard)
        return task_id

    @staticmethod
    async def get_eval_task(redis, task_id: str) -> dict:
        raw = await redis.get(_task_key(task_id))
        if raw is None:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "评测任务不存在或已过期")
        return {"task_id": task_id, **json.loads(raw)}

    @staticmethod
    async def _execute_task(
        task_id: str, redis, agent_id: int, trigger_type: str, operator_id: int
    ) -> None:
        try:
            async with get_db_session() as db:
                result = await EvalService._execute_regression(
                    db, redis, agent_id, trigger_type, operator_id, task_id
                )
        except Exception as exc:
            logger.warning("评测任务执行失败 task_id=%s: %s", task_id, exc, exc_info=True)
            await _update_task(redis, task_id, status="failed", error=str(exc))
            return
        await _update_task(redis, task_id, status="succeeded", result=result)

    @staticmethod
    async def run_regression(
        db: AsyncSession,
        redis,
        agent_id: int,
        trigger_type: str = "publish",
    ) -> dict:
        """同步跑回归评测集并返回门禁判定（发布门禁契约：需要即时结论）。

        评测对象为最新草稿快照（即将生效的配置，见评测后端实现 §1.1）。
        """
        return await EvalService._execute_regression(
            db, redis, agent_id, trigger_type, get_current_user_id()
        )

    @staticmethod
    async def _execute_regression(
        db: AsyncSession,
        redis,
        agent_id: int,
        trigger_type: str,
        operator_id: int | None,
        task_id: str | None = None,
    ) -> dict:
        await EvalService._get_agent_or_raise(db, agent_id)
        dataset = await ai_agent_eval_dataset_repository.get_by_agent_and_type(
            db, agent_id, "regression"
        )
        # 新 Agent 首发无回归集时平凡放行，避免严格阻断造成发布死锁；
        # 后续配置了回归集后，发布门禁按实际评测结果判定。
        if not dataset:
            return _gate_result(passed=True)

        samples = await ai_agent_eval_sample_repository.list_by_dataset(db, dataset.id)
        # 回归集已配置但无考题：门禁无可判依据，阻断发布而非静默放行
        if not samples:
            return _gate_result(passed=False, insufficient_eval=True)

        # 草稿快照：整批样本基于同一配置（即将生效）评测
        version = await ai_agent_version_repository.get_latest_draft(db, agent_id)
        if version is None:
            raise BusinessException(
                ResultCode.RESOURCE_NOT_FOUND, "该 Agent 暂无草稿版本，请先保存草稿"
            )
        snapshot = ai_agent_version_repository.resolve_snapshot(version.snapshot or {})

        run = SysAiAgentEvalRun(
            agent_id=agent_id,
            dataset_id=dataset.id,
            trigger_type=trigger_type,
            status=1,
            create_by=operator_id,
        )
        run = await ai_agent_eval_run_repository.create(db, run)

        total = len(samples)
        results: list[dict] = []
        try:
            for done, sample in enumerate(samples, start=1):
                try:
                    result = await eval_runner.run_sample(db, redis, sample, snapshot)
                except Exception as exc:
                    logger.warning("评测样本 %s 执行异常: %s", sample.id, exc, exc_info=True)
                    result = {
                        "sample_id": sample.id,
                        "task_goal": sample.task_goal,
                        "risk_level": sample.risk_level,
                        "passed": False,
                        "error": str(exc),
                        "actual_output": None,
                        "judge_model": JUDGE_MODEL,
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
                if task_id:
                    await _update_task(
                        redis, task_id, status="running", progress={"done": done, "total": total}
                    )
                # 失败快停：高风险样本失败即终止后续样本（阻断结论已明确，节省评测成本）
                if sample.risk_level == "high" and not result["passed"]:
                    break
        except Exception:
            # 评测中断（DB/Redis 故障）：落失败态与已得结果，避免 run 停留在执行中
            run.status = 3
            run.results = results
            await db.flush()
            raise

        score_summary = _aggregate_scores(results)
        failed_samples = [r for r in results if not r["passed"]]
        # 门禁一：任一维度低于阈值或 high 风险样本失败（EvalRunner 已判定），有失败样本即阻断
        passed = not failed_samples
        # 门禁二：相对退化——总得分（四维均值）相对上次完成评测下降超阈值即阻断
        degraded = False
        if passed:
            previous = await ai_agent_eval_run_repository.get_previous_completed(
                db, agent_id, dataset.id, run.id
            )
            regression_threshold = await get_dict_int(
                db, DICT_TYPE_AI_EVAL, "regression_threshold", REGRESSION_THRESHOLD_DEFAULT
            )
            degraded = _is_degraded(
                _total_score(score_summary),
                _total_score(previous.score_summary) if previous else None,
                regression_threshold,
            )
            passed = not degraded
        run.status = 2 if passed else 3
        run.score_summary = score_summary
        run.results = results
        await db.flush()
        await db.refresh(run)

        return _gate_result(
            passed=passed,
            degraded=degraded,
            score_summary=score_summary,
            failed_samples=failed_samples,
            run_id=run.id,
        )

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


def _gate_result(
    passed: bool,
    degraded: bool = False,
    insufficient_eval: bool = False,
    score_summary: dict | None = None,
    failed_samples: list[dict] | None = None,
    run_id: int | None = None,
) -> dict[str, Any]:
    """门禁判定结果（字段恒定，未评测/样本不足场景也给出完整形状）。"""
    return {
        "passed": passed,
        "degraded": degraded,
        "insufficient_eval": insufficient_eval,
        "score_summary": score_summary or {},
        "failed_samples": failed_samples or [],
        "run_id": run_id,
    }


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
