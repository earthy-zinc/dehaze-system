"""DehazeChatModel 单次调用状态的并发隔离（usage/路由归因/绑定工具）

模型实例按 (agent_id, version_no, model_id) 缓存跨会话复用，单次调用的状态若存
实例字段，并发 run 会互相覆盖——计费与成本归因会串到其他用户。
"""

import asyncio

import pytest
from langchain_core.messages import HumanMessage

from app.infrastructure.llm.client import dehaze_chat_model as m
from app.infrastructure.llm.client.dehaze_chat_model import DehazeChatModel
from tests.stubs.fakes import LLMChunk, NullDBSession


@pytest.fixture(autouse=True)
def _no_db(monkeypatch):
    monkeypatch.setattr(m, "get_db_session", lambda: NullDBSession())


def _patch_stream(monkeypatch, usages, captured, delays=None):
    """按调用顺序返回 usage 的 stream_chat 桩（delays 控制各次调用完成时机）。"""
    calls = {"n": 0}

    async def _stream_chat(db, model_id, messages, **kwargs):
        idx = calls["n"]
        calls["n"] += 1
        captured.append({"tools": kwargs.get("tools")})
        if kwargs.get("on_route_result"):
            kwargs["on_route_result"]({"model_id": f"model-{idx}"})
        yield LLMChunk(type="text_delta", content="回答")
        if delays:
            await asyncio.sleep(delays[idx])
        yield LLMChunk(type="done", usage=usages[idx])

    monkeypatch.setattr(m.llm_client, "stream_chat", _stream_chat)


async def test_stream_usage_isolated_between_concurrent_runs(monkeypatch):
    """并发流式 run 各自持有自己的 usage（direct 路径读 model._last_usage）。

    调用 0 先完成、调用 1 后完成：若 usage 存实例字段，先完成的 run 在让出
    事件循环后会读到后完成 run 的 usage（计费串到他用户）。
    """
    captured: list[dict] = []
    _patch_stream(monkeypatch, [{"input_tokens": 11}, {"input_tokens": 22}], captured, [0.01, 0.03])
    model = DehazeChatModel(model="m1")

    async def run(delay: float):
        async for _chunk in model.astream([HumanMessage(content="hi")]):
            pass
        immediate = dict(model._last_usage)
        await asyncio.sleep(delay)
        return immediate, dict(model._last_usage)

    first, second = await asyncio.gather(run(0.08), run(0.01))

    assert first == ({"input_tokens": 11}, {"input_tokens": 11})
    assert second == ({"input_tokens": 22}, {"input_tokens": 22})


async def test_ainvoke_returns_own_usage_under_concurrency(monkeypatch):
    captured: list[dict] = []
    _patch_stream(monkeypatch, [{"input_tokens": 11}, {"input_tokens": 22}], captured, [0.03, 0.01])
    model = DehazeChatModel(model="m1")

    async def run():
        result = await model.ainvoke([HumanMessage(content="hi")])
        return result.response_metadata["usage"]["input_tokens"]

    first, second = await asyncio.gather(run(), run())

    assert {first, second} == {11, 22}


async def test_call_meta_read_from_last_call_of_same_task(monkeypatch):
    captured: list[dict] = []
    _patch_stream(monkeypatch, [{"input_tokens": 1}, {"input_tokens": 2}], captured, [0.01, 0.01])
    model = DehazeChatModel(model="m1")

    async for _chunk in model.astream([HumanMessage(content="hi")]):
        pass
    assert model._last_call_meta == {"model_id": "model-0"}
    async for _chunk in model.astream([HumanMessage(content="hi")]):
        pass
    assert model._last_call_meta == {"model_id": "model-1"}


async def test_bound_tools_not_leaked_to_other_run(monkeypatch):
    """bind_tools 只对本 run 生效（未按 task 隔离会串给未绑定工具的并发 run）。"""
    captured: list[dict] = []
    _patch_stream(monkeypatch, [{"input_tokens": 1}, {"input_tokens": 2}], captured, [0.02, 0.01])
    model = DehazeChatModel(model="m1")

    async def bound_run():
        model.bind_tools([{"type": "function", "function": {"name": "search"}}])
        await asyncio.sleep(0.02)
        await model.ainvoke([HumanMessage(content="hi")])

    async def plain_run():
        await model.ainvoke([HumanMessage(content="hi")])

    await asyncio.gather(bound_run(), plain_run())

    plain, bound = captured[0], captured[1]
    assert plain["tools"] is None
    assert [t["function"]["name"] for t in bound["tools"]] == ["search"]
