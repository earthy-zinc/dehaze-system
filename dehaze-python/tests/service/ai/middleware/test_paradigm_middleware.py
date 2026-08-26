from langchain_core.messages import AIMessage, HumanMessage

from app.service.ai.middleware.paradigm_middleware import ParadigmMiddleware, _PlanExecutor
from app.service.ai.paradigms import plan_execute


def _task(tid, description="任务", depends_on=None, paradigm="react"):
    return {
        "id": tid,
        "description": description,
        "depends_on": depends_on or [],
        "paradigm": paradigm,
        "status": "pending",
        "result": None,
    }


def _plan(*tasks):
    return {"tasks": list(tasks), "status": "executing", "revisions": []}


async def _build_plan_stub(task, model_call, tool_hint=None):
    return {"tasks": [_task("A")], "status": "pending", "revisions": []}


def _collect_emit(events):
    async def _emit(event):
        events.append(event)

    return _emit


def _model_call_returns(returns: dict):
    async def _call(messages, system_prompt):
        content = messages[0]["content"]
        for tid, text in returns.items():
            if content.startswith(f"[{tid}]") or tid in content:
                return text
        return "默认结果"

    return _call


class _Runtime:
    def __init__(self):
        self.events = []

    def stream_writer(self, event):
        self.events.append(event)


class _FakeModel:
    async def ainvoke(self, messages):
        return AIMessage(content="模型输出")


class _IH:
    async def save_interrupt(self, thread_id, itype, data):
        return None


async def test_executor_runs_dependency_batches():
    executor = _PlanExecutor(
        model_call=_model_call_returns({"A": "ra", "B": "rb", "C": "rc", "D": "rd", "E": "re"}),
        max_parallel=4,
        emit=_collect_emit([]),
        reflexion_cfg={},
        ctx={"token_budget": 0},
    )
    plan = _plan(
        _task("A"),
        _task("C"),
        _task("D"),
        _task("B", depends_on=["A"]),
        _task("E", depends_on=["B", "D"]),
    )
    await executor.run(plan)
    status = {t["id"]: t["status"] for t in plan["tasks"]}
    assert status == {"A": "done", "B": "done", "C": "done", "D": "done", "E": "done"}
    assert plan["status"] == "done"


async def test_executor_failure_triggers_replanner():
    events = []
    executor = _PlanExecutor(
        model_call=_model_call_returns({"A": "ra"}),
        max_parallel=2,
        emit=_collect_emit(events),
        reflexion_cfg={},
        ctx={"token_budget": 0},
    )
    plan = _plan(_task("A", description="任务A"), _task("B", description="任务B"))

    async def model_call(messages, system_prompt):
        content = messages[0]["content"]
        if content.startswith("失败子任务"):
            return '{"revised": [{"id": "A2", "description": "修订"}]}'
        if "任务B" in content:
            raise RuntimeError("boom")
        return "ra"

    executor.model_call = model_call
    await executor.run(plan)
    assert len(plan["revisions"]) == 1
    assert plan["revisions"][0]["reason"] == "B"
    assert any(e.get("data", {}).get("phase") == "revised" for e in events)


async def test_executor_precharge_failure_degrades_batch(monkeypatch):
    from app.service.ai.strategies.quota_recall import quota_recall

    async def _precharge(ctx, batch_size):
        return False

    monkeypatch.setattr(quota_recall, "precharge_batch", _precharge)

    executor = _PlanExecutor(
        model_call=_model_call_returns({"A": "ra"}),
        max_parallel=1,
        emit=lambda event: None,
        reflexion_cfg={},
        ctx={},
    )
    plan = _plan(_task("A"))
    await executor.run(plan)
    assert plan["tasks"][0]["status"] == "failed"
    assert "配额不足" in plan["tasks"][0]["result"]


async def test_executor_reflexion_subtask_uses_evaluator(monkeypatch):
    async def model_call(messages, system_prompt):
        return '{"score": 0.9, "feedback": "达标"}'

    executor = _PlanExecutor(
        model_call=model_call,
        max_parallel=1,
        emit=lambda event: None,
        reflexion_cfg={"max_iterations_reflexion": 2, "reflexion_threshold": 0.8},
        ctx={},
    )
    plan = _plan(_task("A", paradigm="reflexion"))
    await executor.run(plan)
    assert plan["tasks"][0]["status"] == "done"
    assert plan["tasks"][0]["result"]


async def test_plan_execute_emits_via_sync_stream_writer(monkeypatch):
    monkeypatch.setattr(plan_execute, "build_plan", _build_plan_stub)
    monkeypatch.setattr("app.service.ai.middleware.paradigm_middleware.interrupt_handler", _IH())
    monkeypatch.setattr("app.service.ai.middleware.paradigm_middleware.interrupt", lambda data: {})

    runtime = _Runtime()
    mw = ParadigmMiddleware(model=_FakeModel(), config={}, ctx={})
    state = {
        "reasoning_mode": "plan_execute",
        "message_id": 5,
        "conversation_id": 1,
        "user_id": 10,
        "model_id": "m1",
        "stream_session_id": "s1",
        "messages": [HumanMessage(content="测试任务")],
    }
    result = await mw._run_plan_execute(state, runtime)

    phases = [e.get("data", {}).get("phase") for e in runtime.events if e.get("type") == "plan"]
    assert phases == ["plan", "approved", "done"]
    assert result.update["final_response"]
    assert result.goto == "end"
    assert not any(k.startswith("_plan_") for k in mw.ctx)
