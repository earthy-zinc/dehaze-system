import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from app.infrastructure.llm.client.dehaze_chat_model import DehazeChatModel
from app.service.ai.middleware.paradigm_middleware import (
    ParadigmMiddleware,
    _ParadigmBlocked,
    _ParadigmUsage,
    _PlanExecutor,
)
from app.service.ai.paradigms import plan_execute
from app.service.ai.strategies.quota_recall import quota_recall


def _ctx(**overrides):
    ctx = {
        "max_steps": 20,
        "token_budget": 100000,
        "token_used": 0,
        "step_count": 0,
        "tool_timeout": 60,
        "retry_max": 2,
    }
    ctx.update(overrides)
    return ctx


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


class _FakeModel(DehazeChatModel):
    """按调用次数返回固定 usage 的模型桩（usage 挂在 response_metadata）。"""

    model: str = "m1"
    content: str = "模型输出"
    usage: dict = Field(default_factory=lambda: {"input_tokens": 10, "output_tokens": 5})
    calls: int = 0

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=self.content,
                        response_metadata={
                            "usage": dict(self.usage),
                            "call_meta": {"model_id": "m1", "request_id": "r1"},
                        },
                    )
                )
            ]
        )


class _IH:
    def __init__(self, interrupt=None):
        self.interrupt = interrupt
        self.saved = []

    async def get_interrupt(self, thread_id):
        return self.interrupt

    async def save_interrupt(self, thread_id, itype, data):
        self.saved.append((thread_id, itype, data))


class _FailingIH:
    async def get_interrupt(self, thread_id):
        return None

    async def save_interrupt(self, thread_id, itype, data):
        raise RuntimeError("redis down")


def _state(**overrides):
    state = {
        "reasoning_mode": "plan_execute",
        "message_id": 5,
        "conversation_id": 1,
        "user_id": 10,
        "model_id": "m1",
        "stream_session_id": "s1",
        "messages": [HumanMessage(content="测试任务")],
    }
    state.update(overrides)
    return state


def _patch_interrupt(monkeypatch, handler=None, resume=None):
    monkeypatch.setattr(
        "app.service.ai.middleware.paradigm_middleware.interrupt_handler",
        handler or _IH(),
    )
    monkeypatch.setattr(
        "app.service.ai.middleware.paradigm_middleware.interrupt",
        lambda data: resume if resume is not None else {},
    )


def _phases(runtime):
    return [e.get("data", {}).get("phase") for e in runtime.events if e.get("type") == "plan"]


# ── 编排出口与计费闭环 ──────────────────────────────────


async def test_plan_execute_jumps_to_end_with_aggregated_usage(monkeypatch):
    """范式内多次 LLM 调用的 usage 汇总到最终消息，且跳到 after_agent 结算。"""
    _patch_interrupt(monkeypatch)
    runtime = _Runtime()
    model = _FakeModel()
    mw = ParadigmMiddleware(model=model, config={}, ctx=_ctx())

    result = await mw.abefore_agent(_state(), runtime)
    assert result is not None

    assert result["jump_to"] == "end"
    # Planner 1 次 + 子任务 1 次
    assert model.calls == 2
    assert result["messages"][0].response_metadata["usage"] == {
        "input_tokens": 20,
        "output_tokens": 10,
        "cached_input_tokens": 0,
    }
    assert result["messages"][0].response_metadata["call_meta"]["model_id"] == "m1"
    assert result["stop_reason"] == "stop"
    assert _phases(runtime) == ["plan", "approved", "done"]


async def test_reflexion_aggregates_usage_of_all_rounds(monkeypatch):
    runtime = _Runtime()
    model = _FakeModel(content='{"score": 0.9, "feedback": "达标"}')
    mw = ParadigmMiddleware(model=model, config={"max_iterations_reflexion": 2}, ctx=_ctx())

    result = await mw.abefore_agent(_state(reasoning_mode="reflexion"), runtime)
    assert result is not None

    assert result["jump_to"] == "end"
    # actor + evaluator 各 1 次（首轮达标即返回）
    assert model.calls == 2
    assert result["messages"][0].response_metadata["usage"]["input_tokens"] == 20


async def test_react_mode_not_intervened():
    mw = ParadigmMiddleware(model=_FakeModel(), config={}, ctx=_ctx())
    assert await mw.abefore_agent(_state(reasoning_mode="react"), _Runtime()) is None


# ── 护栏 ─────────────────────────────────────────────


async def test_paradigm_calls_count_steps_and_tokens(monkeypatch):
    """范式直连调用同样累计 step_count/token_used（护栏判定同源）。"""
    _patch_interrupt(monkeypatch)
    ctx = _ctx()
    mw = ParadigmMiddleware(model=_FakeModel(), config={}, ctx=ctx)

    await mw.abefore_agent(_state(), _Runtime())

    assert ctx["step_count"] == 2
    assert ctx["token_used"] == 30


async def test_step_limit_blocks_paradigm(monkeypatch):
    _patch_interrupt(monkeypatch)
    mw = ParadigmMiddleware(model=_FakeModel(), config={}, ctx=_ctx(max_steps=1))

    result = await mw.abefore_agent(_state(), _Runtime())
    assert result is not None

    assert result["stop_reason"] == "max_steps"
    assert "最大推理步数" in result["final_response"]
    assert result["jump_to"] == "end"


async def test_token_budget_blocks_paradigm(monkeypatch):
    _patch_interrupt(monkeypatch)
    mw = ParadigmMiddleware(model=_FakeModel(), config={}, ctx=_ctx(token_budget=5))

    result = await mw.abefore_agent(_state(), _Runtime())
    assert result is not None

    assert result["stop_reason"] == "token_budget_exceeded"
    assert result["jump_to"] == "end"


# ── 计划确认与 resume ─────────────────────────────────


async def test_resume_restores_approved_plan_from_interrupt(monkeypatch):
    """resume（图节点整体重跑）须恢复用户已确认的计划，不得重建。"""
    approved = _plan(_task("A", description="已确认任务"))
    approved["status"] = "pending"
    handler = _IH(
        interrupt={
            "type": "plan_approve",
            "data": {
                "type": "plan_approve",
                "stream_session_id": "s1",
                "data": {"plan": approved},
            },
        }
    )
    _patch_interrupt(monkeypatch, handler=handler, resume={"plan_edit": None})

    def _must_not_build(*args, **kwargs):
        raise AssertionError("resume 不得重建计划")

    monkeypatch.setattr(plan_execute, "build_plan", _must_not_build)
    runtime = _Runtime()
    mw = ParadigmMiddleware(model=_FakeModel(content="子任务结果"), config={}, ctx=_ctx())

    result = await mw.abefore_agent(_state(), runtime)
    assert result is not None

    assert result["plan"]["tasks"][0]["description"] == "已确认任务"
    assert result["plan"]["status"] == "done"
    assert "子任务结果" in result["final_response"]
    # 恢复的计划不再重复推送"新建计划"事件
    assert _phases(runtime) == ["approved", "done"]


async def test_resume_applies_plan_edit_on_restored_plan(monkeypatch):
    approved = _plan(_task("A"), _task("B"))
    approved["status"] = "pending"
    handler = _IH(
        interrupt={
            "type": "plan_approve",
            "data": {
                "type": "plan_approve",
                "stream_session_id": "s1",
                "data": {"plan": approved},
            },
        }
    )
    _patch_interrupt(monkeypatch, handler=handler, resume={"plan_edit": {"remove": ["B"]}})

    mw = ParadigmMiddleware(model=_FakeModel(content="子任务结果"), config={}, ctx=_ctx())
    result = await mw.abefore_agent(_state(), _Runtime())
    assert result is not None

    assert [t["id"] for t in result["plan"]["tasks"]] == ["A"]


async def test_save_interrupt_failure_does_not_fail_run(monkeypatch):
    """中断点持久化失败（Redis 抖动）不得令本轮推理失败。"""
    _patch_interrupt(monkeypatch, handler=_FailingIH())
    mw = ParadigmMiddleware(model=_FakeModel(), config={}, ctx=_ctx())

    result = await mw.abefore_agent(_state(), _Runtime())
    assert result is not None

    assert result["jump_to"] == "end"
    assert result["final_response"]


# ── 执行器 ────────────────────────────────────────────


async def _noop_emit(event):
    return None


def _executor(model_call, **kwargs):
    return _PlanExecutor(
        model_call=model_call,
        max_parallel=kwargs.pop("max_parallel", 1),
        emit=kwargs.pop("emit", _noop_emit),
        reflexion_cfg=kwargs.pop("reflexion_cfg", {}),
        ctx=kwargs.pop("ctx", _ctx()),
        usage=kwargs.pop("usage", _ParadigmUsage()),
    )


async def test_executor_runs_dependency_batches():
    executor = _executor(
        _model_call_returns({"A": "ra", "B": "rb", "C": "rc", "D": "rd", "E": "re"}),
        max_parallel=4,
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


async def test_executor_subtask_context_contains_done_results():
    """子任务上下文须带已完成任务的结论（此前传 tasks_map 导致恒为空）。"""
    seen = []

    async def model_call(messages, system_prompt):
        seen.append(messages[0]["content"])
        return "结果"

    plan = _plan(_task("A", description="任务A"), _task("B", description="任务B", depends_on=["A"]))
    await _executor(model_call, max_parallel=1).run(plan)

    assert "[A] 任务A: 结果" in seen[1]


async def test_executor_failure_triggers_replanner():
    events = []
    executor = _executor(
        _model_call_returns({"A": "ra"}), max_parallel=2, emit=_collect_emit(events)
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
    assert plan["revisions"] == [{"revisionNo": 1, "reason": "B", "changedTaskIds": ["A2"]}]
    assert any(e.get("data", {}).get("phase") == "revised" for e in events)


async def test_executor_runs_replanned_tasks():
    """Replanner 修订出的新任务必须进入本轮执行，不能停在 pending。"""
    plan = _plan(_task("A", description="任务A"))

    async def model_call(messages, system_prompt):
        content = messages[0]["content"]
        if content.startswith("失败子任务"):
            return '{"revised": [{"id": "A2", "description": "修订任务"}]}'
        if "任务A" in content:
            raise RuntimeError("boom")
        return "修订结果"

    await _executor(model_call).run(plan)

    assert [(t["id"], t["status"], t["result"]) for t in plan["tasks"]] == [
        ("A2", "done", "修订结果")
    ]


async def test_executor_replan_loop_is_bounded():
    """修订持续失败时不得无限重规划。"""
    plan = _plan(_task("A", description="任务A"))
    calls = {"replan": 0}

    async def model_call(messages, system_prompt):
        content = messages[0]["content"]
        if content.startswith("失败子任务"):
            calls["replan"] += 1
            return '{"revised": [{"id": "A9", "description": "永远失败"}]}'
        raise RuntimeError("boom")

    await _executor(model_call).run(plan)

    assert calls["replan"] == 5
    # 轮次耗尽后未执行的任务按失败收尾，不得停在 pending
    assert plan["tasks"][0]["status"] == "failed"
    assert plan["tasks"][0]["result"] == "修订次数耗尽，未执行"


async def test_executor_precharge_failure_degrades_batch(monkeypatch):
    async def _precharge(ctx, tasks):
        return None

    monkeypatch.setattr(quota_recall, "precharge_batch", _precharge)
    plan = _plan(_task("A"))

    await _executor(_model_call_returns({"A": "ra"})).run(plan)

    assert plan["tasks"][0]["status"] == "failed"
    assert "配额不足" in plan["tasks"][0]["result"]


async def test_executor_refunds_batch_precharge(monkeypatch):
    """批结束须退回预留（实际消耗由主结算统一扣减，不退即重复扣减）。"""
    reserved = []
    settled = []

    async def _precharge(ctx, tasks):
        reserved.append([t["id"] for t in tasks])
        return 7

    async def _settle(ctx, amount, usage):
        settled.append((amount, dict(usage)))

    monkeypatch.setattr(quota_recall, "precharge_batch", _precharge)
    monkeypatch.setattr(quota_recall, "settle_batch", _settle)
    usage = _ParadigmUsage()
    usage.add({"input_tokens": 3, "output_tokens": 1})

    plan = _plan(_task("A", description="任务A"), _task("B", description="任务B"))
    await _executor(_model_call_returns({"A": "ra", "B": "rb"}), max_parallel=2, usage=usage).run(
        plan
    )

    assert reserved == [["A", "B"]]
    assert [amount for amount, _ in settled] == [7]


async def test_executor_releases_precharge_when_batch_aborted(monkeypatch):
    """批内异常（护栏拦截）退出时同样释放预留，避免真实配额被长期占用。"""
    settled = []

    async def _precharge(ctx, tasks):
        return 7

    async def _settle(ctx, amount, usage):
        settled.append(amount)

    monkeypatch.setattr(quota_recall, "precharge_batch", _precharge)
    monkeypatch.setattr(quota_recall, "settle_batch", _settle)

    async def model_call(messages, system_prompt):
        raise _ParadigmBlocked({"final_response": "已达步数上限", "stop_reason": "max_steps"})

    plan = _plan(_task("A", description="任务A"))
    with pytest.raises(_ParadigmBlocked):
        await _executor(model_call).run(plan)

    assert settled == [7]


async def test_executor_recalls_pending_when_budget_exhausted(monkeypatch):
    """配额召回：未启动任务跳过并推送提示，且提示推送在锁外（不阻塞并行）。"""
    notified = []

    async def _notify(ctx, count):
        notified.append(count)

    monkeypatch.setattr(quota_recall, "notify_partial_skipped", _notify)
    ctx = _ctx(token_used=1000, token_budget=100)
    plan = _plan(_task("A"), _task("B"), _task("C"))

    await _executor(_model_call_returns({"A": "ra"}), max_parallel=1, ctx=ctx).run(plan)

    assert notified == [3]
    assert all(t["status"] == "failed" and "配额不足" in t["result"] for t in plan["tasks"])


async def test_executor_reflexion_subtask_uses_evaluator(monkeypatch):
    async def model_call(messages, system_prompt):
        return '{"score": 0.9, "feedback": "达标"}'

    plan = _plan(_task("A", paradigm="reflexion"))
    executor = _executor(
        model_call, reflexion_cfg={"max_iterations_reflexion": 2, "reflexion_threshold": 0.8}
    )

    await executor.run(plan)

    assert plan["tasks"][0]["status"] == "done"
    assert plan["tasks"][0]["result"]


async def test_plan_execute_emits_via_sync_stream_writer(monkeypatch):
    monkeypatch.setattr(plan_execute, "build_plan", _build_plan_stub)
    _patch_interrupt(monkeypatch)

    runtime = _Runtime()
    mw = ParadigmMiddleware(model=_FakeModel(), config={}, ctx=_ctx())
    result = await mw.abefore_agent(_state(), runtime)
    assert result is not None

    assert _phases(runtime) == ["plan", "approved", "done"]
    assert result["final_response"]
    assert result["jump_to"] == "end"


@pytest.mark.parametrize("mode", ["plan_execute", "reflexion"])
async def test_paradigm_state_carries_plan_only_for_plan_execute(monkeypatch, mode):
    _patch_interrupt(monkeypatch)
    mw = ParadigmMiddleware(model=_FakeModel(), config={}, ctx=_ctx())

    result = await mw.abefore_agent(_state(reasoning_mode=mode), _Runtime())
    assert result is not None

    if mode == "plan_execute":
        assert result["plan"]["status"] == "done"
    else:
        assert "plan" not in result
