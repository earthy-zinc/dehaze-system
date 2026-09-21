"""智能体评测服务测试：回归门禁（草稿快照/失败快停/相对退化/样本不足阻断）、
异步评测任务状态机（pending→running→succeeded/failed 与逐样本进度）、
样本一致性校验、评测集归属校验、删除与启动评测的审计写入。

eval_runner.run_sample 触发完整推理链路，测试中在单例实例上注入桩结果，
门禁逻辑（快照选择/快停/退化判定/状态流转）走真实实现与真实测试库。
异步任务经 _EVAL_TASKS 等待在途 task（不用 sleep 轮询），Redis 走 fakeredis。
"""

import asyncio
from types import SimpleNamespace

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_agent import SysAiAgent
from app.models.entity.sys_ai_agent_eval_dataset import SysAiAgentEvalDataset
from app.models.entity.sys_ai_agent_eval_run import SysAiAgentEvalRun
from app.models.entity.sys_ai_agent_eval_sample import SysAiAgentEvalSample
from app.models.entity.sys_ai_agent_version import SysAiAgentVersion
from app.models.schema.ai_agent import EvalDatasetCreate, EvalSampleCreate
from app.repository.ai_agent_eval_repository import (
    ai_agent_eval_run_repository,
    ai_agent_eval_sample_repository,
)
from app.service.ai.service.eval_runner import eval_runner
from app.service.ai_eval_service import _EVAL_TASKS, EVAL_TASK_TTL, eval_service

pytestmark = pytest.mark.requires_db

DIMS = ("result_quality", "process_compliance", "safety_boundary", "efficiency")


@pytest.fixture(autouse=True)
def captured_audit(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """接管审计写入：测试不触达真实 Mongo，同时供断言读取审计内容。"""
    calls: list[dict] = []
    monkeypatch.setattr(
        "app.service.ai_eval_service.mongo_audit_log_repository",
        SimpleNamespace(create_audit_async=lambda **kwargs: calls.append(kwargs)),
    )
    return calls


def _result(sample_id: int, passed: bool, risk_level: str = "low") -> dict:
    scores = dict.fromkeys(DIMS, 90.0 if passed else 0.0)
    return {
        "sample_id": sample_id,
        "task_goal": f"任务{sample_id}",
        "risk_level": risk_level,
        "passed": passed,
        "error": None if passed else "执行失败",
        "scores": scores,
        "notes": {},
        "metrics": {},
    }


async def _seed_agent_with_regression(db, code: str, risk_levels: list[str]):
    agent = SysAiAgent(agent_code=code, name=f"Agent {code}", model_id="m")
    db.add(agent)
    await db.flush()
    dataset = SysAiAgentEvalDataset(agent_id=agent.id, name="回归集", dataset_type="regression")
    db.add(dataset)
    await db.flush()
    samples = [
        SysAiAgentEvalSample(dataset_id=dataset.id, task_goal=f"任务{i}", risk_level=rl)
        for i, rl in enumerate(risk_levels, start=1)
    ]
    db.add_all(samples)
    await db.flush()
    return agent, dataset, samples


async def _add_version(db, agent_id: int, version_no: int, status: int, model_id: str):
    version = SysAiAgentVersion(
        agent_id=agent_id,
        version_no=version_no,
        snapshot={"model_id": model_id, "config": {}},
        status=status,
    )
    db.add(version)
    await db.flush()
    return version


async def _add_completed_run(db, agent_id: int, dataset_id: int, total: float) -> SysAiAgentEvalRun:
    run = SysAiAgentEvalRun(
        agent_id=agent_id,
        dataset_id=dataset_id,
        trigger_type="publish",
        status=2,
        score_summary={
            "dimensions": dict.fromkeys(DIMS, total),
            "sample_count": 1,
            "pass_rate": 1.0,
        },
        results=[],
    )
    db.add(run)
    await db.flush()
    return run


async def _drain_eval_tasks() -> None:
    """等待在途评测任务跑完（含其内部独立 DB 会话），避免跨用例残留。"""
    pending = list(_EVAL_TASKS)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


class TestRunRegression:
    async def test_missing_dataset_passes_trivially(self, db):
        agent = SysAiAgent(agent_code="rg_none", name="rg_none", model_id="m")
        db.add(agent)
        await db.flush()
        gate = await eval_service.run_regression(db, None, agent.id, trigger_type="manual")
        assert gate["passed"] is True
        assert gate["run_id"] is None

    async def test_insufficient_samples_blocks_gate(self, db):
        """回归集已配置却无考题：阻断发布而非静默放行，且不落评测记录。"""
        agent, _, _ = await _seed_agent_with_regression(db, "rg_empty", [])
        gate = await eval_service.run_regression(db, None, agent.id, trigger_type="manual")
        assert gate["passed"] is False
        assert gate["insufficient_eval"] is True
        assert gate["run_id"] is None
        _, total = await ai_agent_eval_run_repository.list_by_agent(db, agent.id, 1, 10)
        assert total == 0

    async def test_missing_draft_raises(self, db):
        agent, _, _ = await _seed_agent_with_regression(db, "rg_nodraft", ["low"])
        with pytest.raises(BusinessException) as ei:
            await eval_service.run_regression(db, None, agent.id)
        assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND

    async def test_evaluates_draft_not_published_snapshot(self, db, monkeypatch):
        agent, _, _samples = await _seed_agent_with_regression(db, "rg_draft", ["low"])
        await _add_version(db, agent.id, version_no=1, status=2, model_id="published-m")
        await _add_version(db, agent.id, version_no=2, status=1, model_id="draft-m")
        seen: list[dict] = []

        async def _fake_run_sample(db, redis, sample, snapshot):
            seen.append(snapshot)
            return _result(sample.id, passed=True)

        monkeypatch.setattr(eval_runner, "run_sample", _fake_run_sample)
        gate = await eval_service.run_regression(db, None, agent.id)
        assert gate["passed"] is True
        # 门禁评测对象是即将生效的草稿配置，而非当前已发布版本
        assert seen
        assert seen[0]["model_id"] == "draft-m"

    async def test_fail_fast_stops_after_high_risk_failure(self, db, monkeypatch):
        agent, _, samples = await _seed_agent_with_regression(
            db, "rg_failfast", ["low", "high", "low"]
        )
        await _add_version(db, agent.id, version_no=1, status=1, model_id="draft-m")
        executed: list[int] = []

        async def _fake_run_sample(db, redis, sample, snapshot):
            executed.append(sample.id)
            return _result(
                sample.id, passed=sample.risk_level != "high", risk_level=sample.risk_level
            )

        monkeypatch.setattr(eval_runner, "run_sample", _fake_run_sample)
        gate = await eval_service.run_regression(db, None, agent.id)
        assert gate["passed"] is False
        assert executed == [samples[0].id, samples[1].id]  # 高风险失败后快停
        assert len(gate["failed_samples"]) == 1
        assert gate["failed_samples"][0]["risk_level"] == "high"

    async def test_degradation_over_threshold_blocks(self, db, monkeypatch):
        agent, dataset, _ = await _seed_agent_with_regression(db, "rg_degrade", ["low"])
        await _add_version(db, agent.id, version_no=1, status=1, model_id="draft-m")
        # 上次完成评测总分 90，本次全通过但总分 60：下降 33% > 阈值 5%
        await _add_completed_run(db, agent.id, dataset.id, total=90.0)

        async def _fake_run_sample(db, redis, sample, snapshot):
            result = _result(sample.id, passed=True)
            result["scores"] = dict.fromkeys(DIMS, 60.0)
            return result

        monkeypatch.setattr(eval_runner, "run_sample", _fake_run_sample)
        gate = await eval_service.run_regression(db, None, agent.id)
        assert gate["passed"] is False
        assert gate["degraded"] is True
        assert gate["failed_samples"] == []

    async def test_stable_score_passes(self, db, monkeypatch):
        agent, dataset, _ = await _seed_agent_with_regression(db, "rg_stable", ["low"])
        await _add_version(db, agent.id, version_no=1, status=1, model_id="draft-m")
        await _add_completed_run(db, agent.id, dataset.id, total=90.0)

        async def _fake_run_sample(db, redis, sample, snapshot):
            return _result(sample.id, passed=True)

        monkeypatch.setattr(eval_runner, "run_sample", _fake_run_sample)
        gate = await eval_service.run_regression(db, None, agent.id)
        assert gate["passed"] is True
        assert gate["degraded"] is False


class TestAsyncEvalTask:
    """异步评测任务：受理即返回 task_id，状态与进度落 Redis，任务在后台跑完落库。"""

    async def test_state_machine_and_progress(self, db, mock_redis, monkeypatch):
        agent, _, _ = await _seed_agent_with_regression(db, "task_ok", ["low", "low"])
        await _add_version(db, agent.id, version_no=1, status=1, model_id="draft-m")

        first_entered = asyncio.Event()
        second_entered = asyncio.Event()
        release_first = asyncio.Event()
        release_second = asyncio.Event()
        executed: list[int] = []

        async def _fake_run_sample(db, redis, sample, snapshot):
            executed.append(sample.id)
            # 卡住样本执行，逐档观察 Redis 中的任务状态
            if len(executed) == 1:
                first_entered.set()
                await release_first.wait()
            elif len(executed) == 2:
                second_entered.set()
                await release_second.wait()
            return _result(sample.id, passed=True)

        monkeypatch.setattr(eval_runner, "run_sample", _fake_run_sample)
        task_id = await eval_service.start_eval_task(
            db, mock_redis, agent.id, operator_id=7, trigger_type="manual"
        )

        await asyncio.wait_for(first_entered.wait(), timeout=5)
        accepted = await eval_service.get_eval_task(mock_redis, task_id)
        assert accepted["status"] == "pending"
        assert accepted["progress"] == {"done": 0, "total": 0}

        release_first.set()
        await asyncio.wait_for(second_entered.wait(), timeout=5)
        running = await eval_service.get_eval_task(mock_redis, task_id)
        assert running["status"] == "running"
        assert running["progress"] == {"done": 1, "total": 2}

        release_second.set()
        await _drain_eval_tasks()
        final = await eval_service.get_eval_task(mock_redis, task_id)
        assert final["status"] == "succeeded"
        assert final["progress"] == {"done": 2, "total": 2}
        assert final["error"] is None
        assert final["result"]["passed"] is True
        assert final["result"]["run_id"] is not None

        runs, total = await ai_agent_eval_run_repository.list_by_agent(db, agent.id, 1, 10)
        assert total == 1
        assert runs[0].status == 2
        assert runs[0].results is not None
        assert len(runs[0].results) == 2

        ttl = await mock_redis.ttl(f"ai:eval:task:{task_id}")
        assert 0 < ttl <= EVAL_TASK_TTL

    async def test_failure_records_error_and_failed_status(self, db, mock_redis):
        """执行期异常（无草稿快照）落 failed 与原因，任务不停留在执行中。"""
        agent, _, _ = await _seed_agent_with_regression(db, "task_fail", ["low"])
        task_id = await eval_service.start_eval_task(db, mock_redis, agent.id, operator_id=7)
        await _drain_eval_tasks()

        state = await eval_service.get_eval_task(mock_redis, task_id)
        assert state["status"] == "failed"
        assert "草稿" in state["error"]

    async def test_unknown_task_not_found(self, mock_redis):
        with pytest.raises(BusinessException) as ei:
            await eval_service.get_eval_task(mock_redis, "no-such-task")
        assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


class TestEvalAudit:
    async def test_delete_dataset_writes_audit(self, db, captured_audit):
        agent, dataset, _ = await _seed_agent_with_regression(db, "aud_ds", [])
        await eval_service.delete_dataset(db, agent.id, dataset.id, operator_id=7)
        assert len(captured_audit) == 1
        audit = captured_audit[0]
        assert audit["operator_id"] == 7
        assert audit["target_type"] == "ai_eval_dataset"
        assert audit["target_id"] == dataset.id
        assert audit["action"] == "delete"
        assert audit["before_value"]["agent_id"] == agent.id

    async def test_delete_sample_writes_audit(self, db, captured_audit):
        agent, dataset, samples = await _seed_agent_with_regression(db, "aud_smp", ["low"])
        await eval_service.delete_sample(db, agent.id, samples[0].id, operator_id=7)
        assert len(captured_audit) == 1
        audit = captured_audit[0]
        assert audit["target_type"] == "ai_eval_sample"
        assert audit["target_id"] == samples[0].id
        assert audit["action"] == "delete"
        assert audit["before_value"]["dataset_id"] == dataset.id

    async def test_start_eval_task_writes_audit(self, db, mock_redis, captured_audit):
        agent = SysAiAgent(agent_code="aud_run", name="aud_run", model_id="m")
        db.add(agent)
        await db.flush()
        await eval_service.start_eval_task(
            db, mock_redis, agent.id, operator_id=7, trigger_type="manual"
        )
        await _drain_eval_tasks()
        assert len(captured_audit) == 1
        audit = captured_audit[0]
        assert audit["operator_id"] == 7
        assert audit["target_type"] == "ai_eval_run"
        assert audit["action"] == "start"
        assert audit["after_value"] == {"agent_id": agent.id, "trigger_type": "manual"}


class TestCreateSampleConsistency:
    async def test_dataset_id_mismatch_rejected(self, db):
        agent, dataset, _ = await _seed_agent_with_regression(db, "smp_mismatch", [])
        form = EvalSampleCreate(dataset_id=dataset.id + 1, task_goal="任务")
        with pytest.raises(BusinessException) as ei:
            await eval_service.create_sample(db, agent.id, dataset.id, form)
        assert ei.value.code == ResultCode.PARAM_ERROR

    async def test_dataset_id_match_creates_sample(self, db):
        agent, dataset, _ = await _seed_agent_with_regression(db, "smp_match", [])
        form = EvalSampleCreate(dataset_id=dataset.id, task_goal="任务", risk_level="high")
        sample = await eval_service.create_sample(db, agent.id, dataset.id, form)
        assert sample.dataset_id == dataset.id
        assert sample.risk_level == "high"


class TestDatasetOwnership:
    async def _seed_two_agents(self, db, code_a: str, code_b: str):
        agent_a = SysAiAgent(agent_code=code_a, name=code_a, model_id="m")
        agent_b = SysAiAgent(agent_code=code_b, name=code_b, model_id="m")
        db.add_all([agent_a, agent_b])
        await db.flush()
        dataset_a = SysAiAgentEvalDataset(agent_id=agent_a.id, name="A集", dataset_type="dev")
        dataset_b = SysAiAgentEvalDataset(agent_id=agent_b.id, name="B集", dataset_type="dev")
        db.add_all([dataset_a, dataset_b])
        await db.flush()
        sample_a = SysAiAgentEvalSample(dataset_id=dataset_a.id, task_goal="A 任务")
        db.add(sample_a)
        await db.flush()
        return agent_a, agent_b, dataset_a, dataset_b, sample_a

    async def test_cross_agent_dataset_update_rejected(self, db):
        _agent_a, agent_b, dataset_a, _, _ = await self._seed_two_agents(db, "own_ua", "own_ub")
        with pytest.raises(BusinessException) as ei:
            await eval_service.update_dataset(db, agent_b.id, dataset_a.id, _form(name="x"))
        assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND

    async def test_cross_agent_dataset_delete_rejected(self, db):
        _agent_a, agent_b, dataset_a, _, _ = await self._seed_two_agents(db, "own_da", "own_db")
        with pytest.raises(BusinessException) as ei:
            await eval_service.delete_dataset(db, agent_b.id, dataset_a.id, operator_id=7)
        assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND

    async def test_cross_agent_sample_update_rejected(self, db):
        _agent_a, agent_b, _, _, sample_a = await self._seed_two_agents(db, "own_sa", "own_sb")
        with pytest.raises(BusinessException) as ei:
            await eval_service.update_sample(db, agent_b.id, sample_a.id, _form(task_goal="x"))
        assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND

    async def test_delete_dataset_cascades_samples(self, db):
        agent_a, _, dataset_a, _, sample_a = await self._seed_two_agents(db, "own_ca", "own_cb")
        await eval_service.delete_dataset(db, agent_a.id, dataset_a.id, operator_id=7)
        assert await ai_agent_eval_sample_repository.get_by_id(db, sample_a.id) is None


class _Form:
    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def model_dump(self, exclude_unset=True):
        return self._kwargs


def _form(**kwargs) -> _Form:
    return _Form(**kwargs)


class TestDatasetRecreateAfterDelete:
    """唯一键 (agent_id, dataset_type) 含软删行：删除后重建同类型复活原行，活跃行才拒绝。"""

    async def _seed_agent(self, db, code: str) -> SysAiAgent:
        agent = SysAiAgent(agent_code=code, name=f"Agent {code}", model_id="m")
        db.add(agent)
        await db.flush()
        return agent

    async def test_recreate_same_type_after_delete_revives_row(self, db):
        agent = await self._seed_agent(db, "recreate_a1")
        dataset = await eval_service.create_dataset(
            db, agent.id, EvalDatasetCreate(name="开发集", description="", dataset_type="dev")
        )
        await eval_service.delete_dataset(db, agent.id, dataset.id, operator_id=7)

        recreated = await eval_service.create_dataset(
            db, agent.id, EvalDatasetCreate(name="开发集v2", description="重建", dataset_type="dev")
        )

        assert recreated.id == dataset.id
        assert recreated.deleted == 0
        assert recreated.name == "开发集v2"

    async def test_create_same_type_when_alive_raises(self, db):
        agent = await self._seed_agent(db, "recreate_a2")
        await eval_service.create_dataset(
            db, agent.id, EvalDatasetCreate(name="回归集", dataset_type="regression")
        )
        with pytest.raises(BusinessException) as ei:
            await eval_service.create_dataset(
                db, agent.id, EvalDatasetCreate(name="回归集2", dataset_type="regression")
            )
        assert ei.value.code == ResultCode.DATA_EXISTS
