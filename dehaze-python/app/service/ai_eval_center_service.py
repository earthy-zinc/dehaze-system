"""评测中心聚合服务（EvalCenterService）

跨 Agent 评测聚合：总览（各 Agent 最近得分/门禁状态/退化标识）、历史趋势、
两次 run 对比、判分模型状态（人工复核一致率推导漂移）、人工复核队列与回填。

判分模型状态为轻量实现：不做在线验证器，从人工复核一致率 + sys_dict 阈值
（ai_eval.judge_consistency_threshold）推导一致性状态与门禁暂停提示。

人工复核抽样规则：失败样本全量纳入（门禁/高风险校准价值最高），通过样本按
sys_dict ai_eval.judge_review_ratio（百分比）确定性抽样，(run_id, sample_id)
唯一键幂等生成，避免重复入队。
"""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_agent_eval_run import SysAiAgentEvalRun
from app.models.entity.sys_ai_eval_review import SysAiEvalReview
from app.repository.ai_agent_eval_repository import (
    ai_agent_eval_review_repository,
    ai_agent_eval_run_repository,
)
from app.repository.ai_agent_repository import ai_agent_repository
from app.service.dict_service import get_dict_int

# sys_dict 读取失败的回退默认值（与 config/sql/data/sys_dict.sql ai_eval 种子一致）
REGRESSION_THRESHOLD_DEFAULT = 5
CONSISTENCY_THRESHOLD_DEFAULT = 90
REVIEW_RATIO_DEFAULT = 1

# 复核队列扫描的最近已完成评测数与复核明细聚合上限（纯技术参数）
REVIEW_SCAN_RUN_LIMIT = 50
REVIEW_AGGREGATE_LIMIT = 1000

DICT_TYPE_AI_EVAL = "ai_eval"


class EvalCenterService:
    @staticmethod
    async def overview(db: AsyncSession) -> list[dict[str, Any]]:
        agents = await ai_agent_repository.get_all(db)
        runs = await ai_agent_eval_run_repository.list_latest_per_agent(db, per_agent=2)
        runs_by_agent: dict[int, list[SysAiAgentEvalRun]] = {}
        for run in runs:
            runs_by_agent.setdefault(run.agent_id, []).append(run)

        regression_threshold = await get_dict_int(
            db, DICT_TYPE_AI_EVAL, "regression_threshold", REGRESSION_THRESHOLD_DEFAULT
        )
        items = []
        for agent in agents:
            agent_runs = runs_by_agent.get(agent.id, [])
            latest = agent_runs[0] if agent_runs else None
            previous = agent_runs[1] if len(agent_runs) > 1 else None
            total = _total_score(latest.score_summary) if latest else None
            items.append(
                {
                    "agent_id": agent.id,
                    "agent_code": agent.agent_code,
                    "agent_name": agent.name,
                    "run_id": latest.id if latest else None,
                    "run_time": latest.create_time if latest else None,
                    "trigger_type": latest.trigger_type if latest else None,
                    "gate_status": "none" if latest is None else ("passed" if latest.status == 2 else "failed"),
                    "total_score": total,
                    "dimensions": (latest.score_summary or {}).get("dimensions") if latest else None,
                    "degraded": _is_degraded(
                        total, _total_score(previous.score_summary) if previous else None, regression_threshold
                    ),
                    "high_risk_failed": _has_high_risk_failed(latest),
                }
            )
        return items

    @staticmethod
    async def trends(
        db: AsyncSession,
        agent_id: int | None = None,
        start_time=None,
        end_time=None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        runs = await ai_agent_eval_run_repository.list_completed(
            db, agent_id=agent_id, start_time=start_time, end_time=end_time, limit=limit
        )
        agents = {
            a.id: a for a in await ai_agent_repository.get_by_ids(db, list({r.agent_id for r in runs}))
        }
        return [
            {
                "run_id": run.id,
                "agent_id": run.agent_id,
                "agent_name": agents[run.agent_id].name if run.agent_id in agents else None,
                "trigger_type": run.trigger_type,
                "status": run.status,
                "total_score": _total_score(run.score_summary),
                "dimensions": (run.score_summary or {}).get("dimensions"),
                "create_time": run.create_time,
            }
            for run in runs
        ]

    @staticmethod
    async def compare_runs(db: AsyncSession, run_id: int, base_run_id: int) -> dict[str, Any]:
        run = await ai_agent_eval_run_repository.get_by_id(db, run_id)
        base = await ai_agent_eval_run_repository.get_by_id(db, base_run_id)
        if run is None or base is None:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "评测记录不存在")
        if run.agent_id != base.agent_id:
            raise BusinessException(ResultCode.PARAM_ERROR, "两次评测不属于同一 Agent，无法对比")

        return {
            "run_id": run.id,
            "base_run_id": base.id,
            "agent_id": run.agent_id,
            "current": _run_snapshot(run),
            "base": _run_snapshot(base),
            "dimension_diff": _dimension_diff(run.score_summary, base.score_summary),
            "sample_diff": _sample_diff(run.results or [], base.results or []),
        }

    @staticmethod
    async def judge_status(db: AsyncSession) -> dict[str, Any]:
        threshold = await get_dict_int(
            db, DICT_TYPE_AI_EVAL, "judge_consistency_threshold", CONSISTENCY_THRESHOLD_DEFAULT
        )
        reviews = await ai_agent_eval_review_repository.list_all(db, REVIEW_AGGREGATE_LIMIT)
        reviewed = [r for r in reviews if r.status == 2]
        agree_count = sum(1 for r in reviewed if r.agree == 1)
        disagree_count = len(reviewed) - agree_count
        stats = {
            "total": len(reviews),
            "pending": len(reviews) - len(reviewed),
            "reviewed": len(reviewed),
            "agree_count": agree_count,
            "disagree_count": disagree_count,
            "agreement_rate": round(agree_count / len(reviewed) * 100, 2) if reviewed else 0.0,
        }
        if not reviewed:
            state, drift_paused = "insufficient_data", False
        else:
            state = "normal" if stats["agreement_rate"] >= threshold else "drifted"
            drift_paused = state == "drifted"
        return {
            "consistency_state": state,
            "drift_paused": drift_paused,
            "consistency_threshold": threshold,
            "review_stats": stats,
        }

    @staticmethod
    async def list_reviews(db: AsyncSession, status: int | None = None) -> dict[str, Any]:
        """复核队列：先按抽样规则幂等补齐最近评测的待复核项，再返回队列与统计。"""
        runs = await ai_agent_eval_run_repository.list_completed(db, limit=REVIEW_SCAN_RUN_LIMIT)
        await EvalCenterService._materialize_reviews(db, runs)

        reviews = await ai_agent_eval_review_repository.list_all(db, REVIEW_AGGREGATE_LIMIT)
        if status is not None:
            reviews = [r for r in reviews if r.status == status]
        agents = {
            a.id: a
            for a in await ai_agent_repository.get_by_ids(db, list({r.agent_id for r in reviews}))
        }
        items = [
            {
                "id": review.id,
                "run_id": review.run_id,
                "sample_id": review.sample_id,
                "agent_id": review.agent_id,
                "agent_name": agents[review.agent_id].name if review.agent_id in agents else None,
                "judge_passed": bool(review.judge_passed),
                "risk_level": review.risk_level,
                "status": review.status,
                "agree": None if review.agree is None else bool(review.agree),
                "remark": review.remark,
                "create_time": review.create_time,
            }
            for review in reviews
        ]
        return {
            "items": items,
            "pending": sum(1 for r in reviews if r.status == 1),
            "reviewed": sum(1 for r in reviews if r.status == 2),
        }

    @staticmethod
    async def submit_review(
        db: AsyncSession, review_id: int, agree: bool, remark: str | None, reviewer_id: int
    ) -> dict[str, Any]:
        review = await ai_agent_eval_review_repository.get_by_id(db, review_id)
        if review is None:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "复核项不存在")
        if review.status == 2:
            raise BusinessException(ResultCode.OPERATION_NOT_ALLOW, "该复核项已完成复核，不允许重复回填")

        review.agree = 1 if agree else 0
        review.status = 2
        review.reviewer_id = reviewer_id
        review.remark = remark
        await db.flush()
        await db.refresh(review)
        return {
            "id": review.id,
            "run_id": review.run_id,
            "sample_id": review.sample_id,
            "agent_id": review.agent_id,
            "judge_passed": bool(review.judge_passed),
            "risk_level": review.risk_level,
            "status": review.status,
            "agree": bool(review.agree),
            "remark": review.remark,
        }

    @staticmethod
    async def _materialize_reviews(db: AsyncSession, runs: list[SysAiAgentEvalRun]) -> None:
        """按抽样规则为最近评测生成待复核项：(run_id, sample_id) 唯一，幂等。"""
        if not runs:
            return
        ratio = await get_dict_int(db, DICT_TYPE_AI_EVAL, "judge_review_ratio", REVIEW_RATIO_DEFAULT)
        existing = {
            (r.run_id, r.sample_id)
            for r in await ai_agent_eval_review_repository.list_by_run_ids(db, [run.id for run in runs])
        }
        for run in runs:
            for result in run.results or []:
                sample_id = result.get("sample_id")
                if sample_id is None:
                    continue
                passed = bool(result.get("passed"))
                if passed and not _sample_hit(run.id, sample_id, ratio):
                    continue
                if (run.id, sample_id) in existing:
                    continue
                db.add(
                    SysAiEvalReview(
                        run_id=run.id,
                        sample_id=sample_id,
                        agent_id=run.agent_id,
                        judge_passed=1 if passed else 0,
                        risk_level=result.get("risk_level") or "low",
                        status=1,
                    )
                )
                existing.add((run.id, sample_id))
        await db.flush()


def _sample_hit(run_id: int, sample_id: int, ratio: int) -> bool:
    """确定性抽样：ratio 为百分比（0-100），同一 (run_id, sample_id) 结果恒定。"""
    return (run_id * 1_000_003 + sample_id) % 100 < ratio


def _total_score(score_summary: dict[str, Any] | None) -> float | None:
    dimensions = (score_summary or {}).get("dimensions")
    if not dimensions:
        return None
    return round(sum(dimensions.values()) / len(dimensions), 2)


def _is_degraded(current: float | None, previous: float | None, threshold: int) -> bool:
    """退化判定：相对上次评测总分下降超过阈值（百分比）。"""
    if current is None or previous is None or previous <= 0:
        return False
    return (previous - current) / previous * 100 > threshold


def _has_high_risk_failed(run: SysAiAgentEvalRun | None) -> bool:
    if run is None:
        return False
    return any(
        r.get("risk_level") == "high" and not r.get("passed") for r in run.results or []
    )


def _run_snapshot(run: SysAiAgentEvalRun) -> dict[str, Any]:
    summary = run.score_summary or {}
    return {
        "run_id": run.id,
        "total_score": _total_score(summary),
        "dimensions": summary.get("dimensions"),
        "sample_count": summary.get("sample_count", 0),
        "pass_rate": summary.get("pass_rate"),
        "create_time": run.create_time,
    }


def _dimension_diff(
    current: dict[str, Any] | None, base: dict[str, Any] | None
) -> dict[str, float]:
    cur_dims = (current or {}).get("dimensions") or {}
    base_dims = (base or {}).get("dimensions") or {}
    return {
        dim: round(cur_dims.get(dim, 0) - base_dims.get(dim, 0), 2)
        for dim in ("result_quality", "process_compliance", "safety_boundary", "efficiency")
    }


def _sample_diff(current: list[dict], base: list[dict]) -> dict[str, Any]:
    cur_map = {r.get("sample_id"): r for r in current if r.get("sample_id") is not None}
    base_map = {r.get("sample_id"): r for r in base if r.get("sample_id") is not None}

    def _item(sample_id: int, result: dict, base_result: dict | None) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "task_goal": result.get("task_goal", ""),
            "current_passed": result.get("passed"),
            "base_passed": base_result.get("passed") if base_result else None,
            "current_score": _sample_total(result),
            "base_score": _sample_total(base_result) if base_result else None,
            "score_delta": (
                round(_sample_total(result) - _sample_total(base_result), 2)
                if base_result
                else None
            ),
        }

    added = [_item(sid, cur_map[sid], None) for sid in cur_map if sid not in base_map]
    removed = [_item(sid, base_map[sid], None) for sid in base_map if sid not in cur_map]
    changed, unchanged = [], 0
    for sid in cur_map:
        if sid not in base_map:
            continue
        cur_total, base_total = _sample_total(cur_map[sid]), _sample_total(base_map[sid])
        if cur_map[sid].get("passed") != base_map[sid].get("passed") or cur_total != base_total:
            changed.append(_item(sid, cur_map[sid], base_map[sid]))
        else:
            unchanged += 1
    return {"added": added, "removed": removed, "changed": changed, "unchanged_count": unchanged}


def _sample_total(result: dict) -> float | None:
    scores = result.get("scores")
    if not scores:
        return None
    return round(sum(scores.values()) / len(scores), 2)


eval_center_service = EvalCenterService()
