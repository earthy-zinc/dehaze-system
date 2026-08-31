"""可观测性采集链路测试：上下文快照、LLM 调用明细、trace 聚合落盘与旁路容错"""

from types import SimpleNamespace

import pytest

from app.service.ai.service import trace_collector
from app.service.ai.service.trace_collector import TraceCollector, error_type_of
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.repository.ai_llm_call_repository import ai_llm_call_repository
from app.repository.ai_trace_repository import ai_trace_repository

pytestmark = pytest.mark.requires_db


@pytest.fixture(autouse=True)
def _reset_collector():
    """每个用例结束后重置 ContextVar，避免采集器跨用例残留"""
    yield
    trace_collector._current_collector.set(None)


def _chunk(type: str, content: str = "", usage: dict | None = None, name: str = ""):
    return SimpleNamespace(
        type=type, content=content, usage=usage, tool_call_id="c1", tool_call_name=name
    )


def _collector(conv_id: int = 1, model: str = "gpt-x") -> TraceCollector:
    return trace_collector.start(
        conversation_id=conv_id,
        message_id=11,
        user_id=42,
        agent_code="default",
        model_id=model,
    )


# ── 上下文快照（§2.2）─────────────────────────────────


def test_record_context_snapshot_composition():
    collector = _collector()
    collector.record_context(
        system_prompt="S" * 40,
        messages=[
            {"role": "system", "content": "摘要块"},  # system 注入不计入历史
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "tool", "content": "result"},
        ],
        injected_memories=[
            {"memory_id": 1, "memory_type": "preference", "source": "manual", "content": "mem-1"},
            {"memory_id": 2, "memory_type": "fact", "source": "auto", "content": "mem-2"},
        ],
        summary="早期对话摘要",
    )
    types = {item["type"]: item for item in collector.context_items}
    assert set(types) == {"system", "summary", "history", "memory"}
    assert types["history"]["counts"] == {"user": 1, "assistant": 1, "tool": 1}
    assert types["history"]["source"] == "raw"
    assert types["memory"]["count"] == 2
    assert types["memory"]["items"][0] == {
        "memory_id": 1,
        "memory_type": "preference",
        "source": "manual",
        "content": "mem-1",
    }
    assert types["summary"]["content"] == "早期对话摘要"
    assert collector.context_events == [
        {"event": "summarize", "tokens": types["summary"]["tokens"]}
    ]


def test_system_prompt_content_truncated_to_limit():
    collector = _collector()
    collector.record_context(
        system_prompt="x" * (trace_collector.SYSTEM_PROMPT_MAX_CHARS + 100),
        messages=[],
        injected_memories=[],
        summary=None,
    )
    item = next(i for i in collector.context_items if i["type"] == "system")
    assert len(item["content"]) == trace_collector.SYSTEM_PROMPT_MAX_CHARS


# ── LLM 调用明细（§2.3）───────────────────────────────


async def test_llm_call_record_finish_writes_row(db):
    collector = _collector()
    tools = [{"name": "search", "description": "搜索"}]
    call = collector.begin_llm_call(
        "gpt-x", [{"role": "user", "content": "hi"}], "sys", tools
    )
    call.observe_chunk(_chunk("text_delta", "Hel"))
    call.observe_chunk(_chunk("tool_call_complete", '{"q":1}', name="search"))
    call.observe_chunk(
        _chunk("done", usage={"prompt_tokens": 10, "completion_tokens": 5, "cached_tokens": 4})
    )
    await call.finish(completed=True)

    calls = await ai_llm_call_repository.list_by_trace(db, collector.trace_id)
    assert len(calls) == 1
    row = calls[0]
    assert row.seq == 1 and row.status == 1 and row.model == "gpt-x"
    assert row.prompt_tokens == 10 and row.completion_tokens == 5 and row.cached_tokens == 4
    assert row.tool_call == {"has_tool_call": True, "tools": [{"name": "search", "arguments": '{"q":1}'}]}
    assert row.input_snapshot["messages"]["counts"] == {"user": 1, "assistant": 0, "tool": 0, "system": 0}
    assert row.input_snapshot["messages"]["items"] == [{"role": "user", "content": "hi"}]
    assert row.input_snapshot["tools"] == [{"name": "search", "description": "搜索"}]
    assert row.input_snapshot["tool_count"] == 1
    assert row.input_snapshot["system_tokens"] == 1  # 估算口径 max(1, len//4)
    assert row.input_snapshot["system_content"] == "sys"
    assert row.input_snapshot["user_id"] == 42
    assert row.output_snapshot["text"] == "Hel"
    assert collector.llm_call_count == 1
    assert collector.first_token_ms is not None


async def test_input_snapshot_truncates_long_content(db):
    collector = _collector()
    long_msg = "m" * (trace_collector.MESSAGE_CONTENT_MAX_CHARS + 100)
    long_desc = "d" * (trace_collector.TOOL_DESC_MAX_CHARS + 100)
    call = collector.begin_llm_call(
        "gpt-x",
        [{"role": "user", "content": long_msg}],
        None,
        [{"name": "search", "description": long_desc}],
    )
    await call.finish(completed=True)
    calls = await ai_llm_call_repository.list_by_trace(db, collector.trace_id)
    snapshot = calls[0].input_snapshot
    assert snapshot["messages"]["items"] == [
        {"role": "user", "content": "m" * trace_collector.MESSAGE_CONTENT_MAX_CHARS}
    ]
    assert snapshot["tools"] == [{"name": "search", "description": "d" * trace_collector.TOOL_DESC_MAX_CHARS}]


async def test_memory_items_recorded_with_content(db):
    collector = _collector()
    long_mem = "c" * (trace_collector.MEMORY_CONTENT_MAX_CHARS + 100)
    collector.record_context(
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
        injected_memories=[
            {"memory_id": 7, "memory_type": "fact", "source": "auto", "content": long_mem}
        ],
        summary=None,
    )
    await collector.settle(status=trace_collector.TRACE_STATUS_SUCCESS)
    trace = await ai_trace_repository.get_by_trace_id(db, collector.trace_id)
    memory_item = next(
        i for i in trace.context_snapshot["items"] if i["type"] == "memory"
    )
    assert memory_item["count"] == 1
    assert memory_item["items"] == [
        {
            "memory_id": 7,
            "memory_type": "fact",
            "source": "auto",
            "content": "c" * trace_collector.MEMORY_CONTENT_MAX_CHARS,
        }
    ]


async def test_llm_call_finish_truncates_output_to_500_chars(db):
    collector = _collector()
    call = collector.begin_llm_call("gpt-x", [], None, None)
    call.observe_chunk(_chunk("text_delta", "x" * 900))
    await call.finish(completed=True)
    calls = await ai_llm_call_repository.list_by_trace(db, collector.trace_id)
    assert len(calls[0].output_snapshot["text"]) == trace_collector.OUTPUT_SNAPSHOT_MAX_CHARS


async def test_llm_call_finish_failed_bypass(db, monkeypatch):
    """采集旁路硬要求：写库失败不影响调用链路，仅告警"""
    collector = _collector()
    call = collector.begin_llm_call("gpt-x", [], None, None)

    async def _boom(db, values):
        raise RuntimeError("db down")

    monkeypatch.setattr(ai_llm_call_repository, "insert_idempotent", _boom)
    await call.finish(completed=False, error_type="5xx")  # 不抛错
    assert collector.llm_call_count == 1


async def test_llm_call_finish_timeout_status(db):
    collector = _collector()
    call = collector.begin_llm_call("gpt-x", [], None, None)
    await call.finish(completed=False, error_type="timeout")
    calls = await ai_llm_call_repository.list_by_trace(db, collector.trace_id)
    assert calls[0].status == 3


async def test_begin_llm_call_skipped_when_no_collector_or_settled():
    assert trace_collector.begin_llm_call("m", [], None, None) is None  # 无采集器
    collector = _collector()
    collector._settled = True
    assert trace_collector.begin_llm_call("m", [], None, None) is None  # 已结算（bypass_span 外的旁路调用）


# ── 物理调用尝试明细（B1：逐 Key/逐路由）──────────────


async def test_llm_call_attempts_persisted(db):
    """Key 重试/路由切换的多次物理尝试逐条落入 attempts"""
    collector = _collector()
    call = collector.begin_llm_call("gpt-x", [], None, None)
    call.observe_attempt(
        provider_id=3, key_id=None, model="gpt-x", status=2, error_code="no_key", latency_ms=None
    )
    call.observe_attempt(
        provider_id=3, key_id=9, model="gpt-x", status=1, error_code=None, latency_ms=120
    )
    await call.finish(completed=True)
    calls = await ai_llm_call_repository.list_by_trace(db, collector.trace_id)
    assert calls[0].attempts == [
        {
            "provider_id": 3,
            "key_id": None,
            "model": "gpt-x",
            "status": 2,
            "error_code": "no_key",
            "latency_ms": None,
        },
        {
            "provider_id": 3,
            "key_id": 9,
            "model": "gpt-x",
            "status": 1,
            "error_code": None,
            "latency_ms": 120,
        },
    ]


async def test_llm_call_attempts_none_when_unobserved(db):
    collector = _collector()
    call = collector.begin_llm_call("gpt-x", [], None, None)
    await call.finish(completed=True)
    calls = await ai_llm_call_repository.list_by_trace(db, collector.trace_id)
    assert calls[0].attempts is None


# ── 旁路独立过程链（bypass_span）──────────────────────


async def test_bypass_span_records_independent_trace(db):
    """bypass_span 内 LLM 调用产出独立 trace（trace_type 正确、聚合口径）"""
    async with trace_collector.bypass_span(
        conversation_id=2, message_id=None, user_id=42, model_id="gpt-x", trace_type="summary"
    ):
        bypass_trace_id = trace_collector.current().trace_id
        call = trace_collector.begin_llm_call("gpt-x", [], None, None)
        assert call is not None
        call.observe_chunk(_chunk("done", usage={"prompt_tokens": 7, "completion_tokens": 3}))
        await call.finish(completed=True)
    assert trace_collector.current() is None  # 退出后 ContextVar 清理
    trace = await ai_trace_repository.get_by_trace_id(db, bypass_trace_id)
    assert trace.trace_type == "summary"
    assert trace.message_id is None
    assert trace.llm_call_count == 1
    assert trace.prompt_tokens == 7 and trace.completion_tokens == 3 and trace.total_tokens == 10


async def test_bypass_span_restores_outer_collector(db):
    main = _collector()
    async with trace_collector.bypass_span(
        conversation_id=2,
        message_id=None,
        user_id=None,
        model_id="m",
        trace_type="memory_extraction",
    ):
        assert trace_collector.current() is not main
        assert trace_collector.current().trace_type == "memory_extraction"
    assert trace_collector.current() is main


async def test_bypass_span_failure_finalizes_then_reraises(db):
    """异常路径：落失败态后原样抛出（调用方兜底仍生效），ContextVar 已清理"""
    trace_id = None
    with pytest.raises(RuntimeError, match="boom"):
        async with trace_collector.bypass_span(
            conversation_id=1, message_id=None, user_id=None, model_id="m", trace_type="suggestion"
        ):
            trace_id = trace_collector.current().trace_id
            raise RuntimeError("boom")
    assert trace_collector.current() is None
    trace = await ai_trace_repository.get_by_trace_id(db, trace_id)
    assert trace.trace_type == "suggestion"
    assert trace.status == trace_collector.TRACE_STATUS_FAILED
    assert trace.error_type == "RuntimeError"


async def test_main_trace_defaults_to_conversation_type(db):
    collector = _collector()
    await collector.settle(status=trace_collector.TRACE_STATUS_SUCCESS)
    trace = await ai_trace_repository.get_by_trace_id(db, collector.trace_id)
    assert trace.trace_type == "conversation"


# ── 消息级聚合落盘（§2.4）─────────────────────────────


async def test_settle_writes_trace_with_billing_usage(db):
    collector = _collector()
    collector.record_context(system_prompt="sys", messages=[{"role": "user", "content": "hi"}],
                             injected_memories=[], summary=None)
    await collector.settle(
        status=trace_collector.TRACE_STATUS_SUCCESS,
        usage={"input_tokens": 100, "output_tokens": 50, "cached_input_tokens": 20},
        step_count=3,
        actual_model="fallback-model",
    )
    trace = await ai_trace_repository.get_by_trace_id(db, collector.trace_id)
    assert trace.status == 1
    assert trace.model == "fallback-model"  # 实际路由归因（降级场景）
    assert trace.total_tokens == 150 and trace.cached_tokens == 20
    assert trace.step_count == 3 and trace.llm_call_count == 0
    assert trace.first_token_ms is None
    assert {i["type"] for i in trace.context_snapshot["items"]} == {"system", "history"}


async def test_settle_idempotent_by_trace_id(db):
    collector = _collector()
    await collector.settle(status=trace_collector.TRACE_STATUS_SUCCESS, step_count=1)
    await collector.settle(status=trace_collector.TRACE_STATUS_FAILED, error_type="x")  # 重复结算跳过
    traces, total = await ai_trace_repository.list_traces(db, conversation_id=1)
    assert total == 1 and traces[0].status == 1


async def test_settle_fallback_to_llm_call_aggregation(db):
    """计费 usage 缺失时回退 LLM 调用聚合口径"""
    collector = _collector()
    call = collector.begin_llm_call("gpt-x", [], None, None)
    call.observe_chunk(_chunk("done", usage={"prompt_tokens": 7, "completion_tokens": 3}))
    await call.finish(completed=True)
    await collector.settle(status=trace_collector.TRACE_STATUS_SUCCESS)
    trace = await ai_trace_repository.get_by_trace_id(db, collector.trace_id)
    assert trace.llm_call_count == 1
    assert trace.prompt_tokens == 7 and trace.completion_tokens == 3 and trace.total_tokens == 10


# ── 异常映射与 finalize 辅助 ──────────────────────────


def test_error_type_of():
    assert error_type_of(BusinessException(ResultCode.AI_LLM_CALL_FAILED, "x")) == "A0600"
    assert error_type_of(TimeoutError()) == "TimeoutError"


async def test_finalize_unsettled_writes_failure(db):
    _collector()
    await trace_collector.finalize_unsettled(
        status=trace_collector.TRACE_STATUS_INTERRUPTED
    )
    traces, total = await ai_trace_repository.list_traces(db, conversation_id=1)
    assert total == 1 and traces[0].status == 3


async def test_finalize_unsettled_writes_error_detail(db):
    _collector()
    detail = {"message": "boom", "stack": "Traceback ..."}
    await trace_collector.finalize_unsettled(
        status=trace_collector.TRACE_STATUS_FAILED,
        error_type="A0600",
        error_detail=detail,
    )
    traces, total = await ai_trace_repository.list_traces(db, conversation_id=1)
    assert total == 1
    assert traces[0].error_detail == detail


async def test_record_event_writes_context_events(db):
    collector = _collector()
    collector.record_event(event="guardrail", rule="prompt_injection", detail="命中关键词")
    collector.record_event(event="plan", phase="plan", plan_summary='{"tasks":[]}')
    await collector.settle(status=trace_collector.TRACE_STATUS_SUCCESS)
    trace = await ai_trace_repository.get_by_trace_id(db, collector.trace_id)
    assert {"event": "guardrail", "rule": "prompt_injection", "detail": "命中关键词"} in trace.context_snapshot["events"]
    assert {"event": "plan", "phase": "plan", "plan_summary": '{"tasks":[]}'} in trace.context_snapshot["events"]


async def test_record_event_skipped_after_settle():
    collector = _collector()
    collector._settled = True  # 已结算（不触发落库，无 db fixture 不得走真实引擎）
    collector.record_event(event="guardrail", rule="x", detail="y")  # 已结算不写入
    assert collector.context_events == []


async def test_finalize_helpers_noop_without_collector():
    await trace_collector.finalize_unsettled(status=2)  # 无采集器不抛错
    await trace_collector.finalize_success()
