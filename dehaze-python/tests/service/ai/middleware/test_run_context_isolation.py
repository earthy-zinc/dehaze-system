"""运行时上下文隔离回归（图缓存复用下防跨 run 串扰）

图按 (agent_id, version_no, model_id) 缓存复用，若运行时上下文（会话标识/计费
上下文/任务状态）跨 run 共享，并发推理会互相污染——用户 A 的回复流会推给用户 B、
计费结算记错消息。本文件锁定：run ctx 由驱动任务经 ensure_run_ctx 预置（图节点
为独立 asyncio 任务，before_agent 节点内 set 无法跨节点传播）、abefore_agent
复用预置 ctx、每 run 计数重置、工具执行读取 run ctx 而非构图模板。
"""

from types import SimpleNamespace

import pytest

from app.service.ai.builders.dehaze_tools_builder import build_business_tools
from app.service.ai.middleware.dehaze_hooks_middleware import DehazeHooksMiddleware
from app.service.ai.middleware.run_context import (
    _run_ctx,
    current_run_ctx,
    ensure_run_ctx,
    get_run_ctx,
    set_run_ctx,
)


@pytest.fixture(autouse=True)
def _no_lifecycle_hooks(monkeypatch):
    """隔离生命周期钩子（计费预扣会开真实 DB session），本文件只测上下文装配"""

    class _NoHooks:
        async def run_hooks(self, point, state):
            return None

    monkeypatch.setattr("app.service.ai.middleware.dehaze_hooks_middleware.agent_hooks", _NoHooks())


@pytest.fixture(autouse=True)
def _reset_run_ctx():
    """每个用例结束后重置 ContextVar，模拟独立 run 驱动任务的干净上下文"""
    yield
    _run_ctx.set(None)


def _template(**overrides):
    ctx = {
        "max_steps": 20,
        "token_budget": 50000,
        "tool_timeout": 60,
        "retry_max": 2,
        "token_used": 0,
        "step_count": 0,
        "_model_id": "test-model",
    }
    ctx.update(overrides)
    return ctx


def _run_state(**overrides):
    state = {
        "messages": [],
        "conversation_id": 1,
        "user_id": 10,
        "message_id": 2,
        "model_id": "test-model",
        "stream_session_id": "s1",
        "step_count": 0,
        "token_used": 0,
    }
    state.update(overrides)
    return state


async def test_ensure_run_ctx_presets_shared_dict():
    """驱动任务预置 run ctx：图内节点（模型/工具/收尾）全部共享同一 dict"""
    template = _template()
    run1 = ensure_run_ctx(template)
    assert run1 is not template
    # 同一 run 内重复调用幂等复用
    assert ensure_run_ctx(template) is run1
    # abefore_agent 复用预置 ctx（不再新建，节点间共享同一 dict）
    mw = DehazeHooksMiddleware(template)
    await mw.abefore_agent(_run_state(), None)
    assert get_run_ctx() is run1
    assert run1["conversation_id"] == 1

    # 新 run（另一驱动任务，ContextVar 干净）：预置独立 ctx，前一 run 不被污染
    _run_ctx.set(None)
    run2 = ensure_run_ctx(template)
    assert run2 is not run1
    await mw.abefore_agent(_run_state(conversation_id=99), None)
    assert run2["conversation_id"] == 99
    assert run2["step_count"] == 0
    assert run1["conversation_id"] == 1
    # 构图模板保持干净
    assert "conversation_id" not in template


async def test_abefore_agent_creates_ctx_without_preset():
    """无预置（单测直调 middleware）时从模板创建，本任务内 self.ctx 可用"""
    template = _template()
    mw = DehazeHooksMiddleware(template)
    await mw.abefore_agent(_run_state(), None)
    run_ctx = get_run_ctx()
    assert run_ctx is not None
    assert run_ctx is not template
    assert run_ctx["conversation_id"] == 1
    assert run_ctx["step_count"] == 0
    assert "conversation_id" not in template


async def test_subagent_run_reuses_main_ctx_and_resets_are_main_only():
    """子 Agent 与主 run 共享同一 ctx dict：标识天然继承，per-run 重置仅主 run 生效

    预置机制下子 Agent 图的节点在主 run 上下文内执行（task 工具派生），复用
    主 run ctx；子 Agent 的 abefore_agent 不得重置主 run 已累计的计数。
    计费隔离由 is_subagent 标记（ContextVar）在结算钩子内区分，不依赖 dict 隔离。
    """
    main_template = _template()
    main_mw = DehazeHooksMiddleware(main_template)
    await main_mw.abefore_agent(_run_state(), None)
    run_ctx = get_run_ctx()
    assert run_ctx is not None
    run_ctx["billing_context"] = {"billing_id": 7}
    run_ctx["token_used"] = 123
    run_ctx["step_count"] = 5

    sub_mw = DehazeHooksMiddleware(_template(max_steps=5, _model_id="sub-model"))
    from app.service.ai.middleware.run_context import enter_subagent_run, exit_subagent_run

    token = enter_subagent_run("task-t1")
    try:
        # 子 Agent state 不含业务字段（deepagents task 工具仅传任务描述）
        await sub_mw.abefore_agent({"messages": []}, None)
    finally:
        exit_subagent_run(token)
    sub_ctx = get_run_ctx()
    assert sub_ctx is not None

    # 共享同一 dict：会话标识继承、主 run 计数不被重置
    assert sub_ctx is run_ctx
    assert sub_ctx["conversation_id"] == 1
    assert sub_ctx["user_id"] == 10
    assert sub_ctx["stream_session_id"] == "s1"
    assert sub_ctx["step_count"] == 5
    assert sub_ctx["token_used"] == 123
    assert sub_ctx["billing_context"] == {"billing_id": 7}


async def test_current_run_ctx_falls_back_to_template():
    """无 run 上下文（单测直调）时回退模板；两者皆无显式报错"""
    template = {"user_id": 1}
    assert current_run_ctx(template) is template
    try:
        current_run_ctx()
        raised = False
    except RuntimeError:
        raised = True
    assert raised


async def test_tool_reads_run_ctx_over_template(monkeypatch):
    """工具执行读取 run 上下文：同图并发场景取到当次 run 的 user_id 而非构图模板"""
    template = _template(
        conversation_id=1, message_id=2, user_id=10, stream_session_id="s1", model_id="m"
    )
    tools = build_business_tools(template)
    captured = {}

    async def fake_retrieve(query, top_k=5, user_id=None):
        captured["user_id"] = user_id
        return []

    monkeypatch.setattr(
        "app.service.ai.builders.dehaze_tools_builder.knowledge_base_client",
        SimpleNamespace(retrieve=fake_retrieve, format_results=lambda r: ""),
    )

    set_run_ctx(dict(template, user_id=20))  # 另一会话的 run
    tool = next(t for t in tools if t.name == "knowledge_base_search")
    await tool.ainvoke({"query": "知识库检索"})
    assert captured["user_id"] == 20

    # 无 run 上下文回退模板（兼容直调语义）
    _run_ctx.set(None)
    await tool.ainvoke({"query": "知识库检索"})
    assert captured["user_id"] == 10
