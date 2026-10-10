"""PRJ-1 回归：plan_execute 范式必须绑定与主图同一套工具。

覆盖：
- 范式子任务的模型请求必须携带工具 schema（此前为零工具，导致「检索知识库」子任务靠模型编造）；
- 模型请求的工具调用要被真实执行并把结果回灌模型（ReAct 一环）；
- 无工具路径的 prompt 必须带能力声明；
- Planner 的 tool_hint 自由文本要收敛到真实工具名。
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field

from app.infrastructure.llm.client import dehaze_chat_model as dcm
from app.service.ai.middleware.paradigm_middleware import (
    _NO_TOOL_CAPABILITY_NOTICE,
    ParadigmMiddleware,
    _ParadigmUsage,
)
from app.service.ai.paradigms import plan_execute


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


def _state(**overrides):
    state = {"message_id": 5, "conversation_id": 1, "user_id": 10, "model_id": "m1"}
    state.update(overrides)
    return state


class _StubTool(BaseTool):
    """最小工具桩：只需 name 与 async ainvoke，满足范式工具执行契约。

    继承 BaseTool 而非鸭子类型：范式工具集按 `Sequence[BaseTool]` 声明（基类
    AgentMiddleware 的契约），鸭子类型桩会让类型门禁报错。
    """

    name: str = "stub"
    description: str = "测试桩工具"
    result: str = "工具结果"
    calls: list[dict] = Field(default_factory=list)

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("测试桩工具仅支持异步调用")

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        return self.result

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """覆盖 ainvoke：范式直传 args dict，桩不做 schema 校验，只记录调用并回固定结果。"""
        self.calls.append(input)
        return self.result


class _ScriptedModel(dcm.DehazeChatModel):
    """按脚本逐次返回 AIMessage 的模型桩，并记录实际收到的 system 消息。"""

    model: str = "m1"
    script: list = Field(default_factory=list)
    calls: int = 0
    seen_system: list = Field(default_factory=list)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        for m in messages:
            if getattr(m, "type", "") == "system":
                self.seen_system.append(str(m.content))
        idx = min(self.calls - 1, len(self.script) - 1)
        return ChatResult(generations=[ChatGeneration(message=self.script[idx])])


async def test_plan_execute_subtask_request_carries_tool_schema():
    """plan_execute 路径下的模型调用必须绑定工具（请求携带工具 schema）。"""
    tool = _StubTool(name="knowledge_base_search", result="库内文档：平台代号=青鸾")
    model = _ScriptedModel(script=[AIMessage(content="答案")])
    mw = ParadigmMiddleware(model=model, config={}, ctx=_ctx(), tools=[tool])

    call = mw._make_model_call(_state(), _ParadigmUsage())
    out = await call(
        [{"role": "user", "content": "检索知识库，查找平台内部代号"}], "你是子任务执行者。"
    )

    bound = dcm._bound_tools_var.get()
    assert bound, "范式子任务调用必须携带工具 schema（PRJ-1 回归）：实际未绑定任何工具"
    assert [t.name for t in bound] == ["knowledge_base_search"], (
        "范式子任务必须携带与主图同一套工具"
    )
    assert out == "答案"


async def test_plan_execute_subtask_executes_tool_call_and_feeds_result_back():
    """模型请求的工具调用要真实执行，结果回灌模型后再产出最终答复。"""
    tool = _StubTool(name="knowledge_base_search", result="值班联系人：王工，分机 8613")
    tool_call = {
        "name": "knowledge_base_search",
        "args": {"query": "值班分机"},
        "id": "call_1",
        "type": "tool_call",
    }
    model = _ScriptedModel(
        script=[
            AIMessage(content="", tool_calls=[tool_call]),
            AIMessage(content="值班联系人分机号是 8613。"),
        ]
    )
    mw = ParadigmMiddleware(model=model, config={}, ctx=_ctx(), tools=[tool])

    call = mw._make_model_call(_state(), _ParadigmUsage())
    out = await call([{"role": "user", "content": "检索知识库，查找值班分机"}], "SYS")

    assert tool.calls == [{"query": "值班分机"}], "工具必须被真实执行一次"
    assert model.calls >= 2, "工具结果必须回灌模型再产出答复"
    assert "8613" in out


async def test_no_tools_path_declares_incapability():
    """未绑定工具时，系统提示必须带「无检索能力」声明，防模型编造。"""
    model = _ScriptedModel(script=[AIMessage(content="无法确认")])
    mw = ParadigmMiddleware(model=model, config={}, ctx=_ctx())

    call = mw._make_model_call(_state(), _ParadigmUsage())
    await call([{"role": "user", "content": "平台代号是什么"}], "SYS")

    assert model.seen_system, "系统提示必须被模型收到（否则能力声明无从体现）"
    assert any(_NO_TOOL_CAPABILITY_NOTICE in s for s in model.seen_system), (
        "未绑定工具的系统提示必须带「无检索能力」声明，防模型编造"
    )


async def test_tools_path_does_not_add_incapability_notice():
    """绑定了工具时不加能力声明（避免自相矛盾）。"""
    model = _ScriptedModel(script=[AIMessage(content="答案")])
    mw = ParadigmMiddleware(
        model=model, config={}, ctx=_ctx(), tools=[_StubTool(name="web_search")]
    )

    call = mw._make_model_call(_state(), _ParadigmUsage())
    await call([{"role": "user", "content": "搜索一下"}], "SYS")

    assert all(_NO_TOOL_CAPABILITY_NOTICE not in s for s in model.seen_system)


@pytest.mark.parametrize(
    ("hint", "available", "expected"),
    [
        ("知识库检索", {"knowledge_base_search"}, "knowledge_base_search"),
        ("knowledge_retrieval", {"knowledge_base_search"}, "knowledge_base_search"),
        ("knowledge_base_search", {"knowledge_base_search"}, "knowledge_base_search"),
        ("none", {"knowledge_base_search"}, None),
        ("无", {"knowledge_base_search"}, None),
        ("", {"knowledge_base_search"}, None),
        (None, {"knowledge_base_search"}, None),
        ("web_search", {"knowledge_base_search"}, None),
        ("知识库检索", set(), None),
        ("知识库检索", None, "knowledge_base_search"),
    ],
)
def test_resolve_tool_hint_whitelist(hint, available, expected):
    """tool_hint 自由文本 → 真实工具名：命中白名单才保留，否则丢弃。"""
    assert plan_execute.resolve_tool_hint(hint, available) == expected


async def test_build_plan_normalizes_tool_hint_and_lists_whitelist():
    """Planner 提示词要列出工具白名单，产物 tool_hint 收敛为真实工具名。"""
    captured: dict = {}

    async def _model_call(messages, system_prompt):
        captured["prompt"] = system_prompt
        return '{"tasks": [{"id": "A", "description": "检索知识库", "tool_hint": "知识库检索"}]}'

    plan = await plan_execute.build_plan(
        "平台代号是什么", _model_call, available_tools=["knowledge_base_search"]
    )

    assert plan["tasks"][0]["tool_hint"] == "knowledge_base_search"
    assert "knowledge_base_search" in captured["prompt"]


async def test_build_plan_drops_unavailable_tool_hint():
    """计划承诺了不存在/未启用的工具时必须丢弃，不向外透传。"""

    async def _model_call(messages, system_prompt):
        return '{"tasks": [{"id": "A", "description": "x", "tool_hint": "自研检索器"}]}'

    plan = await plan_execute.build_plan(
        "随便问问", _model_call, available_tools=["knowledge_base_search"]
    )

    assert plan["tasks"][0]["tool_hint"] is None
