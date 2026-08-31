"""评测中心聚合服务测试：总览/退化判定/趋势/对比/判分状态/人工复核。"""

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_agent import SysAiAgent
from app.models.entity.sys_ai_agent_eval_run import SysAiAgentEvalRun
from app.repository.ai_agent_eval_repository import ai_agent_eval_review_repository
from app.service.ai_eval_center_service import _sample_hit, eval_center_service

pytestmark = pytest.mark.api


DIMS = ("result_quality", "process_compliance", "safety_boundary", "efficiency")


def _summary(result_quality, process_compliance=90, safety_boundary=100, efficiency=80):
    dims = dict(zip(DIMS, (result_quality, process_compliance, safety_boundary, efficiency)))
    return {"dimensions": dims, "sample_count": 1, "pass_rate": 1.0}


def _sample_result(sample_id, passed, risk_level="low", result_quality=90):
    return {
        "sample_id": sample_id,
        "task_goal": f"任务{sample_id}",
        "risk_level": risk_level,
        "passed": passed,
        "scores": dict(zip(DIMS, (result_quality, 90, 100, 80))),
    }


async def _create_agent(db, code: str) -> SysAiAgent:
    agent = SysAiAgent(agent_code=code, name=f"Agent {code}", model_id="test-model")
    db.add(agent)
    await db.flush()
    return agent


async def _create_run(
    db,
    agent_id: int,
    status: int,
    result_quality: float,
    results: list[dict] | None = None,
    trigger_type: str = "manual",
) -> SysAiAgentEvalRun:
    run = SysAiAgentEvalRun(
        agent_id=agent_id,
        dataset_id=1,
        trigger_type=trigger_type,
        status=status,
        score_summary=_summary(result_quality),
        results=results or [],
    )
    db.add(run)
    await db.flush()
    return run


class TestOverview:
    async def test_agent_without_run_gate_none(self, db):
        agent = await _create_agent(db, "ov_none")
        items = await eval_center_service.overview(db)
        item = next(i for i in items if i["agent_id"] == agent.id)
        assert item["gate_status"] == "none"
        assert item["total_score"] is None
        assert item["run_id"] is None
        assert item["degraded"] is False
        assert item["high_risk_failed"] is False

    async def test_gate_passed_with_scores(self, db):
        agent = await _create_agent(db, "ov_pass")
        await _create_run(db, agent.id, status=2, result_quality=80)
        items = await eval_center_service.overview(db)
        item = next(i for i in items if i["agent_id"] == agent.id)
        assert item["gate_status"] == "passed"
        assert item["total_score"] == (80 + 90 + 100 + 80) / 4
        assert item["dimensions"]["result_quality"] == 80
        assert item["degraded"] is False

    async def test_gate_failed(self, db):
        agent = await _create_agent(db, "ov_fail")
        await _create_run(db, agent.id, status=3, result_quality=40)
        items = await eval_center_service.overview(db)
        item = next(i for i in items if i["agent_id"] == agent.id)
        assert item["gate_status"] == "failed"

    async def test_degraded_over_threshold(self, db):
        agent = await _create_agent(db, "ov_degrade")
        # 先创建上次（总分 90），后创建本次（总分 75，下降 16.7% > 阈值 5%）
        await _create_run(db, agent.id, status=2, result_quality=90)
        await _create_run(db, agent.id, status=2, result_quality=60)
        items = await eval_center_service.overview(db)
        item = next(i for i in items if i["agent_id"] == agent.id)
        assert item["degraded"] is True

    async def test_not_degraded_within_threshold(self, db):
        agent = await _create_agent(db, "ov_stable")
        # 先创建上次（总分 90），后创建本次（总分 87.5，下降 2.8% ≤ 5%）
        await _create_run(db, agent.id, status=2, result_quality=90)
        await _create_run(db, agent.id, status=2, result_quality=80)
        items = await eval_center_service.overview(db)
        item = next(i for i in items if i["agent_id"] == agent.id)
        assert item["degraded"] is False

    async def test_high_risk_failed_flag(self, db):
        agent = await _create_agent(db, "ov_highrisk")
        results = [
            _sample_result(1, True),
            _sample_result(2, False, risk_level="high"),
        ]
        await _create_run(db, agent.id, status=3, result_quality=50, results=results)
        items = await eval_center_service.overview(db)
        item = next(i for i in items if i["agent_id"] == agent.id)
        assert item["high_risk_failed"] is True


class TestTrends:
    async def test_order_and_agent_filter(self, db):
        agent_a = await _create_agent(db, "tr_a")
        agent_b = await _create_agent(db, "tr_b")
        await _create_run(db, agent_a.id, status=2, result_quality=80)
        await _create_run(db, agent_b.id, status=3, result_quality=50)
        await _create_run(db, agent_a.id, status=2, result_quality=90)

        agent_items = await eval_center_service.trends(db, agent_id=agent_a.id)
        assert len(agent_items) == 2
        assert all(i["agent_id"] == agent_a.id for i in agent_items)
        assert all(i["agent_name"] == "Agent tr_a" for i in agent_items)


class TestCompare:
    async def test_dimension_and_sample_diff(self, db):
        agent = await _create_agent(db, "cmp_a")
        base_results = [
            _sample_result(1, True, result_quality=90),
            _sample_result(2, True, result_quality=80),
        ]
        cur_results = [
            _sample_result(1, True, result_quality=70),  # 退化
            _sample_result(3, False),  # 新增样本
        ]
        base = await _create_run(db, agent.id, status=2, result_quality=85, results=base_results)
        cur = await _create_run(db, agent.id, status=3, result_quality=70, results=cur_results)

        result = await eval_center_service.compare_runs(db, cur.id, base.id)
        assert result["agent_id"] == agent.id
        assert result["dimension_diff"]["result_quality"] == -15.0
        assert result["current"]["run_id"] == cur.id
        assert result["base"]["run_id"] == base.id
        sample_diff = result["sample_diff"]
        assert {i["sample_id"] for i in sample_diff["added"]} == {3}
        assert {i["sample_id"] for i in sample_diff["removed"]} == {2}
        changed = {i["sample_id"]: i for i in sample_diff["changed"]}
        assert 1 in changed
        assert changed[1]["current_passed"] is True
        assert changed[1]["score_delta"] < 0
        assert sample_diff["unchanged_count"] == 0

    async def test_unchanged_sample_counted(self, db):
        agent = await _create_agent(db, "cmp_same")
        results = [_sample_result(1, True, result_quality=90)]
        base = await _create_run(db, agent.id, status=2, result_quality=90, results=results)
        cur = await _create_run(db, agent.id, status=2, result_quality=90, results=results)
        result = await eval_center_service.compare_runs(db, cur.id, base.id)
        assert result["sample_diff"]["unchanged_count"] == 1
        assert not result["sample_diff"]["changed"]

    async def test_cross_agent_rejected(self, db):
        agent_a = await _create_agent(db, "cmp_x")
        agent_b = await _create_agent(db, "cmp_y")
        run_a = await _create_run(db, agent_a.id, status=2, result_quality=80)
        run_b = await _create_run(db, agent_b.id, status=2, result_quality=80)
        with pytest.raises(BusinessException) as ei:
            await eval_center_service.compare_runs(db, run_a.id, run_b.id)
        assert ei.value.code == ResultCode.PARAM_ERROR

    async def test_missing_run_not_found(self, db):
        with pytest.raises(BusinessException) as ei:
            await eval_center_service.compare_runs(db, 999999, 999998)
        assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


class TestJudgeStatus:
    async def test_insufficient_data_without_review(self, db):
        result = await eval_center_service.judge_status(db)
        assert result["consistency_state"] == "insufficient_data"
        assert result["drift_paused"] is False
        assert result["consistency_threshold"] == 90

    async def test_normal_when_agreement_above_threshold(self, db):
        agent = await _create_agent(db, "js_ok")
        # 失败样本必入复核队列，规避通过样本按比例抽样的不确定性
        results = [_sample_result(1, False), _sample_result(2, False)]
        await _create_run(db, agent.id, status=3, result_quality=50, results=results)
        queue = await eval_center_service.list_reviews(db)
        assert len(queue["items"]) == 2
        for item in queue["items"]:
            await eval_center_service.submit_review(db, item["id"], True, None, 1)

        result = await eval_center_service.judge_status(db)
        assert result["consistency_state"] == "normal"
        assert result["drift_paused"] is False
        assert result["review_stats"]["reviewed"] == 2
        assert result["review_stats"]["agreement_rate"] == 100.0

    async def test_drift_paused_when_disagreement(self, db):
        agent = await _create_agent(db, "js_drift")
        await _create_run(
            db, agent.id, status=3, result_quality=50, results=[_sample_result(1, False)]
        )
        queue = await eval_center_service.list_reviews(db)
        assert len(queue["items"]) == 1
        await eval_center_service.submit_review(db, queue["items"][0]["id"], False, "判分有误", 1)

        result = await eval_center_service.judge_status(db)
        assert result["consistency_state"] == "drifted"
        assert result["drift_paused"] is True
        assert result["review_stats"]["disagree_count"] == 1


class TestReviews:
    async def _run_with_samples(self, db, code: str) -> SysAiAgentEvalRun:
        agent = await _create_agent(db, code)
        results = [_sample_result(i, False) for i in range(1, 3)] + [
            _sample_result(i, True) for i in range(3, 13)
        ]
        return await _create_run(db, agent.id, status=3, result_quality=50, results=results)

    async def test_failed_samples_always_queued(self, db):
        run = await self._run_with_samples(db, "rv_fail")
        queue = await eval_center_service.list_reviews(db)
        queued_ids = {i["sample_id"] for i in queue["items"] if i["run_id"] == run.id}
        assert {1, 2} <= queued_ids
        assert all(i["judge_passed"] is False for i in queue["items"] if i["sample_id"] in (1, 2))

    async def test_materialize_idempotent(self, db):
        await self._run_with_samples(db, "rv_idem")
        first = await eval_center_service.list_reviews(db)
        second = await eval_center_service.list_reviews(db)
        assert len(first["items"]) == len(second["items"])
        assert second["pending"] == first["pending"]

    async def test_status_filter(self, db):
        await self._run_with_samples(db, "rv_filter")
        await eval_center_service.list_reviews(db)
        pending = await ai_agent_eval_review_repository.list_all(db)
        first = pending[0]
        await eval_center_service.submit_review(db, first.id, True, None, 1)
        queue = await eval_center_service.list_reviews(db, status=2)
        assert all(i["status"] == 2 for i in queue["items"])
        assert queue["reviewed"] == 1

    async def test_submit_review_and_reject_duplicate(self, db):
        await self._run_with_samples(db, "rv_submit")
        queue = await eval_center_service.list_reviews(db)
        item = queue["items"][0]
        result = await eval_center_service.submit_review(db, item["id"], False, "误判", 7)
        assert result["status"] == 2
        assert result["agree"] is False
        assert result["remark"] == "误判"

        with pytest.raises(BusinessException) as ei:
            await eval_center_service.submit_review(db, item["id"], True, None, 7)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW

    async def test_submit_unknown_review_not_found(self, db):
        with pytest.raises(BusinessException) as ei:
            await eval_center_service.submit_review(db, 999999, True, None, 1)
        assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


class TestSampleHit:
    def test_boundary_ratios(self):
        assert _sample_hit(1, 1, 100) is True
        assert _sample_hit(1, 1, 0) is False

    def test_deterministic(self):
        assert _sample_hit(42, 7, 50) == _sample_hit(42, 7, 50)

    def test_ratio_one_approximately_one_percent(self):
        hits = sum(1 for n in range(10000) if _sample_hit(1, n, 1))
        assert 30 <= hits <= 170
