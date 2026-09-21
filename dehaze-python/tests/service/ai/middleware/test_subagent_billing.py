"""子 Agent 计费口径：父预扣 + 子实报实销（middleware/钩子级）

口径（AI对话 后端实现-架构与公共 §子 Agent 计费）：
- 主图 run 持有预扣-结算链路（billing_context），行为不变；
- 子图 run（deepagents task 工具作用域内）不预扣，after_agent 按实际用量
  实报实销（settle_subagent），独立计费记录 bill_type=chat_subagent；
- 子图 run 结束后还原 run 上下文与子 run 标记，防主图后续步骤读到子图 ctx。
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from app.service.ai.middleware import agent_hooks as hooks_module
from app.service.ai.middleware import dehaze_hooks_middleware as mw_module
from app.service.ai.middleware.dehaze_hooks_middleware import DehazeHooksMiddleware
from app.service.ai.middleware.run_context import (
    enter_subagent_run,
    exit_subagent_run,
    get_run_ctx,
    in_subagent_run,
    set_run_ctx,
)


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


class _RecordingHooks:
    """记录 before_agent hook_state 的假钩子框架（不触 DB）"""

    def __init__(self):
        self.states = []

    async def run_hooks(self, point, state):
        self.states.append((point, dict(state)))
        return


@pytest.fixture
def recording_hooks(monkeypatch):
    hooks = _RecordingHooks()
    monkeypatch.setattr(mw_module, "agent_hooks", hooks)
    return hooks


class TestSubagentScope:
    async def test_task_tool_marks_subagent_scope_and_restores_ctx(self):
        """task 工具执行期间标记子 run；结束后还原 run 上下文与标记"""
        mw = DehazeHooksMiddleware(_template())
        parent_ctx = _template(conversation_id=1, message_id=2, user_id=10)
        set_run_ctx(parent_ctx)

        captured = {}

        async def handler(request):
            captured["in_scope"] = in_subagent_run()
            # 模拟子图 abefore_agent 覆盖 ContextVar
            set_run_ctx({"spoofed": "sub"})
            return ToolMessage(content="ok", tool_call_id="t1")

        request = SimpleNamespace(tool_call={"id": "t1", "name": "task"})
        await mw.awrap_tool_call(request, handler)

        assert captured["in_scope"] is True
        # 子图覆盖的 ctx 被还原为主 run 上下文，标记退出
        assert get_run_ctx() is parent_ctx
        assert in_subagent_run() is False

    async def test_task_tool_scope_restored_on_error(self, monkeypatch):
        """task 工具执行异常（子图未完成）同样还原标记与上下文，不产生子口径残留"""
        mw = DehazeHooksMiddleware(_template())
        parent_ctx = _template()
        set_run_ctx(parent_ctx)

        async def handler(request):
            enter_subagent_run("task-t1")
            set_run_ctx({"spoofed": "sub"})
            raise RuntimeError("subgraph crashed")

        recovered = []

        async def fake_recover(request, handler, tool_call_id, tool_name, exc):
            recovered.append((tool_name, in_subagent_run()))
            return ToolMessage(content="failed", tool_call_id=tool_call_id)

        monkeypatch.setattr(mw, "_recover_tool", fake_recover)
        request = SimpleNamespace(tool_call={"id": "t1", "name": "task"})
        result = await mw.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        # 恢复流程在还原前执行时仍处于子口径内，但退出后上下文与标记均还原
        assert recovered == [("task", True)]
        assert get_run_ctx() is parent_ctx
        assert in_subagent_run() is False

    async def test_nested_task_scope_restores_layer_by_layer(self):
        """嵌套子图（Team 多级协作）逐层还原：内层退出回到外层子图上下文"""
        outer_ctx = _template(name="outer")
        set_run_ctx(outer_ctx)
        mw = DehazeHooksMiddleware(_template())

        async def outer_handler(request):
            token = enter_subagent_run("task-t1")
            inner_ctx = _template(name="inner")
            set_run_ctx(inner_ctx)
            try:
                assert in_subagent_run() is True
                return ToolMessage(content="inner", tool_call_id="t2")
            finally:
                exit_subagent_run(token)
                set_run_ctx(outer_ctx)

        request = SimpleNamespace(tool_call={"id": "t1", "name": "task"})
        await mw.awrap_tool_call(request, outer_handler)

        assert get_run_ctx() is outer_ctx
        assert in_subagent_run() is False

    async def test_non_task_tool_leaves_scope_untouched(self):
        """普通工具不改变子 run 标记与上下文"""
        mw = DehazeHooksMiddleware(_template())
        set_run_ctx(_template())

        async def handler(request):
            assert in_subagent_run() is False
            return ToolMessage(content="ok", tool_call_id="t1")

        request = SimpleNamespace(tool_call={"id": "t1", "name": "knowledge_base_search"})
        await mw.awrap_tool_call(request, handler)
        assert in_subagent_run() is False


class TestSubagentRunCtx:
    async def test_subagent_run_ctx_marked_and_hook_state_flagged(self, recording_hooks):
        """子 Agent run 的 before_agent hook_state 携带 is_subagent=True（经
        _in_subagent ContextVar 判定，节点任务间天然传播），不携带预扣上下文"""
        token = enter_subagent_run("task-t1")
        try:
            mw = DehazeHooksMiddleware(_template(max_steps=5, _model_id="sub-model"))
            await mw.abefore_agent({"messages": []}, None)
        finally:
            exit_subagent_run(token)

        point, hook_state = recording_hooks.states[0]
        assert point == "before_agent"
        assert hook_state["is_subagent"] is True
        # 子 Agent 不携带预扣上下文（实报实销口径）
        assert hook_state["billing_context"] is None

    async def test_main_run_ctx_not_marked(self, recording_hooks):
        """主图 run is_subagent=False，计费口径不受影响"""
        mw = DehazeHooksMiddleware(_template())
        await mw.abefore_agent(_run_state(), None)

        _, hook_state = recording_hooks.states[0]
        assert hook_state["is_subagent"] is False


def _fake_db_session(monkeypatch, target_module):
    """替换钩子模块的 get_db_session：产出 None 会话（结算服务已被 stub，不触库）"""

    @asynccontextmanager
    async def _fake():
        yield None

    monkeypatch.setattr(target_module, "get_db_session", _fake)


class TestBillingHookRouting:
    @pytest.fixture
    def billing_recorder(self, monkeypatch):
        calls = {"settle": [], "settle_subagent": [], "pre_charge": []}

        async def _pre_charge(db, *args, **kwargs):
            calls["pre_charge"].append(args)
            return {"billing_id": 1}

        async def _settle(db, *args, **kwargs):
            calls["settle"].append(args)

        async def _settle_subagent(db, *args, **kwargs):
            calls["settle_subagent"].append((args, kwargs))

        monkeypatch.setattr(
            hooks_module,
            "billing_service",
            SimpleNamespace(
                pre_charge=_pre_charge, settle=_settle, settle_subagent=_settle_subagent
            ),
        )
        _fake_db_session(monkeypatch, hooks_module)
        return calls

    async def test_precharge_skipped_for_subagent(self, billing_recorder):
        """子 Agent run 不预扣：配额/余额预扣与预估均不发生"""
        result = await hooks_module._billing_pre_charge_hook(
            {
                "is_subagent": True,
                "user_id": 10,
                "conversation_id": 1,
                "message_id": 2,
                "model_id": "sub-model",
                "messages": [],
            }
        )
        assert result is None
        assert billing_recorder["pre_charge"] == []

    async def test_precharge_runs_for_main(self, billing_recorder):
        """主图 run 预扣行为不变"""
        result = await hooks_module._billing_pre_charge_hook(
            {
                "user_id": 10,
                "conversation_id": 1,
                "message_id": 2,
                "model_id": "m1",
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        assert result is None
        assert len(billing_recorder["pre_charge"]) == 1

    async def test_settle_hook_routes_subagent_to_settle_subagent(self, billing_recorder):
        """子图 after_agent 实报实销：走 settle_subagent，归属主会话用户/消息"""
        state = {
            "is_subagent": True,
            "user_id": 10,
            "conversation_id": 1,
            "message_id": 2,
            "model_id": "sub-model",
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "call_meta": {"model_id": "route-fallback", "provider_id": 7},
        }
        assert await hooks_module._billing_settle_hook(state) is None

        assert billing_recorder["settle"] == []
        args, kwargs = billing_recorder["settle_subagent"][0]
        # (user_id, conversation_id, message_id, model_id, actual_model_id, usage)
        assert args[:5] == (10, 1, 2, "sub-model", "route-fallback")
        assert args[5] == {"input_tokens": 100, "output_tokens": 50}
        assert kwargs["provider_id"] == 7

    async def test_settle_hook_subagent_without_identity_skipped(self, billing_recorder):
        """子图缺业务标识（如无消息上下文）时不产生计费记录"""
        state = {"is_subagent": True, "user_id": 10, "usage": {}}
        assert await hooks_module._billing_settle_hook(state) is None
        assert billing_recorder["settle_subagent"] == []

    async def test_settle_hook_main_unchanged(self, billing_recorder):
        """主图 after_agent 差额退补结算链路不变（不重复结算子用量）"""
        state = {
            "user_id": 10,
            "conversation_id": 1,
            "message_id": 2,
            "model_id": "m1",
            "billing_context": {
                "user_id": 10,
                "conversation_id": 1,
                "message_id": 2,
                "billing_id": 7,
            },
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "call_meta": {},
        }
        assert await hooks_module._billing_settle_hook(state) is None
        assert len(billing_recorder["settle"]) == 1
        assert billing_recorder["settle_subagent"] == []

    async def test_trace_settle_skipped_for_subagent(self, monkeypatch):
        """子图 after_agent 不结算主消息过程链（子图调用聚合进主 trace）"""
        touched = []
        monkeypatch.setattr(
            "app.service.ai.service.trace_collector.current",
            lambda: touched.append("current"),
        )
        assert await hooks_module._trace_settle_hook({"is_subagent": True}) is None
        assert touched == []
