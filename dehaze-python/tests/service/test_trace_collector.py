"""可观测性采集链路测试：上下文快照、LLM 调用明细、trace 聚合落盘与旁路容错

trace/llm_call 落盘已后台化（create_task），断言 DB 前须确定性等待后台任务。
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.repository.ai_llm_call_repository import ai_llm_call_repository
from app.repository.ai_trace_repository import ai_trace_repository
from app.service.ai.service import trace_collector
from app.service.ai.service.trace_collector import TraceCollector, error_type_of

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


async def _drain():
    """等待后台落盘任务完成（观测写库不阻塞主链路，断言前需显式等待）"""
    await trace_collector.drain()


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
    call = collector.begin_llm_call("gpt-x", [{"role": "user", "content": "hi"}], "sys", tools)
    call.observe_chunk(_chunk("text_delta", "Hel"))
    call.observe_chunk(_chunk("tool_call_complete", '{"q":1}', name="search"))
    call.observe_chunk(
        _chunk("done", usage={"prompt_tokens": 10, "completion_tokens": 5, "cached_tokens": 4})
    )
    await call.finish(completed=True)
    await _drain()

    calls = await ai_llm_call_repository.list_by_trace(db, collector.trace_id)
    assert len(calls) == 1
    row = calls[0]
    assert row.input_snapshot is not None
    assert row.output_snapshot is not None
    assert row.seq == 1
    assert row.status == 1
    assert row.model == "gpt-x"
    assert row.prompt_tokens == 10
    assert row.completion_tokens == 5
    assert row.cached_tokens == 4
    assert row.tool_call == {
        "has_tool_call": True,
        "tools": [{"name": "search", "arguments": '{"q":1}'}],
    }
    assert row.input_snapshot["messages"]["counts"] == {
        "user": 1,
        "assistant": 0,
        "tool": 0,
        "system": 0,
    }
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
    await _drain()
    calls = await ai_llm_call_repository.list_by_trace(db, collector.trace_id)
    snapshot = calls[0].input_snapshot
    assert snapshot is not None
    assert snapshot["messages"]["items"] == [
        {"role": "user", "content": "m" * trace_collector.MESSAGE_CONTENT_MAX_CHARS}
    ]
    assert snapshot["tools"] == [
        {"name": "search", "description": "d" * trace_collector.TOOL_DESC_MAX_CHARS}
    ]


async def test_input_snapshot_tools_openai_nested_format(db):
    """DehazeChatModel 经 _tools_to_openai 传入 OpenAI function 嵌套格式，
    工具名/描述不得落空（审计需可见每次调用携带的工具清单）"""
    collector = _collector()
    tools = [
        {"type": "function", "function": {"name": "dehaze", "description": "图像去雾"}},
        {"name": "search", "description": "搜索"},
    ]
    call = collector.begin_llm_call("gpt-x", [], None, tools)
    await call.finish(completed=True)
    await _drain()
    calls = await ai_llm_call_repository.list_by_trace(db, collector.trace_id)
    snapshot = calls[0].input_snapshot
    assert snapshot is not None
    assert snapshot["tool_count"] == 2
    assert snapshot["tools"] == [
        {"name": "dehaze", "description": "图像去雾"},
        {"name": "search", "description": "搜索"},
    ]


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
    await _drain()
    trace = await ai_trace_repository.get_by_trace_id(db, collector.trace_id)
    assert trace is not None
    context_snapshot = trace.context_snapshot
    assert context_snapshot is not None
    memory_item = next(i for i in context_snapshot["items"] if i["type"] == "memory")
    assert memory_item["count"] == 1
    assert memory_item["items"] == [
        {
            "memory_id": 7,
            "memory_type": "fact",
            "source": "auto",
            "content": "c" * trace_collector.MEMORY_CONTENT_MAX_CHARS,
        }
    ]


async def test_llm_call_finish_truncates_oversized_output(db):
    """审计级快照存全文；超上限（防病态载荷）时截断"""
    collector = _collector()
    call = collector.begin_llm_call("gpt-x", [], None, None)
    call.observe_chunk(
        _chunk("text_delta", "x" * (trace_collector.OUTPUT_SNAPSHOT_MAX_CHARS + 100))
    )
    await call.finish(completed=True)
    await _drain()
    calls = await ai_llm_call_repository.list_by_trace(db, collector.trace_id)
    output_snapshot = calls[0].output_snapshot
    assert output_snapshot is not None
    assert len(output_snapshot["text"]) == trace_collector.OUTPUT_SNAPSHOT_MAX_CHARS


async def test_llm_call_finish_keeps_full_output_below_limit(db):
    collector = _collector()
    call = collector.begin_llm_call("gpt-x", [], None, None)
    call.observe_chunk(_chunk("text_delta", "完整回复内容"))
    await call.finish(completed=True)
    await _drain()
    calls = await ai_llm_call_repository.list_by_trace(db, collector.trace_id)
    output_snapshot = calls[0].output_snapshot
    assert output_snapshot is not None
    assert output_snapshot["text"] == "完整回复内容"


async def test_llm_call_finish_failed_bypass(db, monkeypatch):
    """采集旁路硬要求：写库失败不影响调用链路，仅告警"""
    collector = _collector()
    call = collector.begin_llm_call("gpt-x", [], None, None)

    async def _boom(db, values):
        raise RuntimeError("db down")

    monkeypatch.setattr(ai_llm_call_repository, "insert_idempotent", _boom)
    await call.finish(completed=False, error_type="5xx")  # 不抛错
    await _drain()  # 后台写库失败仅告警，不得冒泡
    assert collector.llm_call_count == 1


async def test_llm_call_finish_timeout_status(db):
    collector = _collector()
    call = collector.begin_llm_call("gpt-x", [], None, None)
    await call.finish(completed=False, error_type="timeout")
    await _drain()
    calls = await ai_llm_call_repository.list_by_trace(db, collector.trace_id)
    assert calls[0].status == 3


async def test_begin_llm_call_skipped_when_no_collector_or_settled():
    assert trace_collector.begin_llm_call("m", [], None, None) is None  # 无采集器
    collector = _collector()
    collector._settled = True
    assert (
        trace_collector.begin_llm_call("m", [], None, None) is None
    )  # 已结算（bypass_span 外的旁路调用）


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
    await _drain()
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
    await _drain()
    calls = await ai_llm_call_repository.list_by_trace(db, collector.trace_id)
    assert calls[0].attempts is None


# ── wire 级原始报文（审计级重构 P0）───────────────────


async def test_wire_request_response_persisted(db):
    """wire 请求/响应原文与 start_time 落库；start_time 由首次请求上报触发"""
    collector = _collector()
    call = collector.begin_llm_call("gpt-x", [], None, None)
    assert call._start_time is None
    payload = {"model": "gpt-x", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    call.observe_wire_request(payload)
    assert call._start_time is not None
    raw_response = {
        "id": "chatcmpl-1",
        "model": "gpt-x",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "回复"},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    call.observe_wire_response(raw_response)
    await call.finish(completed=True)
    await _drain()
    calls = await ai_llm_call_repository.list_by_trace(db, collector.trace_id)
    row = calls[0]
    assert row.raw_request == payload
    assert row.raw_response == raw_response
    assert row.start_time is not None


async def test_wire_request_start_time_recorded_once(db):
    """Key 重试多次物理尝试共用同一 start_time（首次上报时刻）"""
    collector = _collector()
    call = collector.begin_llm_call("gpt-x", [], None, None)
    call.observe_wire_request({"model": "gpt-x"})
    first = call._start_time
    call.observe_wire_request({"model": "gpt-x", "retry": True})
    assert call._start_time is first
    assert call._raw_request == {"model": "gpt-x"}  # 首次请求体为准


async def test_wire_oversized_truncated_with_annotation(db):
    """raw 超 256KB 时整体替换为截断标注（_original_bytes 为序列化原始字节数）"""
    collector = _collector()
    call = collector.begin_llm_call("gpt-x", [], None, None)
    oversized = {
        "messages": [{"role": "user", "content": "x" * (trace_collector.RAW_WIRE_MAX_BYTES + 1000)}]
    }
    call.observe_wire_request(oversized)
    raw = json.dumps(oversized, ensure_ascii=False)
    assert call._raw_request == {
        "_truncated": True,
        "_original_bytes": len(raw.encode()),
    }
    await call.finish(completed=True)
    await _drain()
    calls = await ai_llm_call_repository.list_by_trace(db, collector.trace_id)
    assert calls[0].raw_request == {"_truncated": True, "_original_bytes": len(raw.encode())}


async def test_wire_error_response_on_all_attempts_failed(db):
    """物理尝试全失败：raw_request 照存，raw_response 为最后错误结构"""
    collector = _collector()
    call = collector.begin_llm_call("gpt-x", [], None, None)
    call.observe_wire_request({"model": "gpt-x", "messages": []})
    call.observe_wire_error("429", "限流")
    call.observe_wire_error("5xx", "服务端错误")  # 最后错误覆盖前次
    await call.finish(completed=False, error_type="5xx")
    await _drain()
    calls = await ai_llm_call_repository.list_by_trace(db, collector.trace_id)
    row = calls[0]
    assert row.status == 2
    assert row.raw_request == {"model": "gpt-x", "messages": []}
    assert row.raw_response == {"error": {"code": "5xx", "message": "服务端错误"}}


async def test_wire_success_overrides_prior_error(db):
    """Key 重试后成功：真实响应覆盖前次失败错误"""
    collector = _collector()
    call = collector.begin_llm_call("gpt-x", [], None, None)
    call.observe_wire_request({"model": "gpt-x"})
    call.observe_wire_error("429", "限流")
    success = {"id": "chatcmpl-2", "choices": [{"finish_reason": "stop", "message": {}}]}
    call.observe_wire_response(success)
    await call.finish(completed=True)
    await _drain()
    calls = await ai_llm_call_repository.list_by_trace(db, collector.trace_id)
    assert calls[0].raw_response == success


async def test_legacy_row_wire_fields_null_without_wire(db):
    """老链路/老数据兼容：无 wire 上报时三字段落 NULL，既有采集不受影响"""
    collector = _collector()
    call = collector.begin_llm_call("gpt-x", [], None, None)
    call.observe_chunk(_chunk("text_delta", "Hel"))
    await call.finish(completed=True)
    await _drain()
    calls = await ai_llm_call_repository.list_by_trace(db, collector.trace_id)
    row = calls[0]
    assert row.start_time is None
    assert row.raw_request is None
    assert row.raw_response is None
    assert row.output_snapshot is not None
    assert row.output_snapshot["text"] == "Hel"  # 摘要视图语义不变


def test_record_wire_helpers_delegate_to_current_wire_record():
    """协议客户端上报通道：wire_record 挂载时经模块级函数透传，未挂载时静默跳过"""
    trace_collector.record_wire_request({"no": "collector"})  # 无挂载不抛错
    collector = _collector()
    call = collector.begin_llm_call("gpt-x", [], None, None)
    token = trace_collector.wire_record.set(call)
    try:
        trace_collector.record_wire_request({"model": "gpt-x"})
        trace_collector.record_wire_response({"id": "chatcmpl-3"})
    finally:
        trace_collector.wire_record.reset(token)
    assert call._raw_request == {"model": "gpt-x"}
    assert call._raw_response == {"id": "chatcmpl-3"}


# ── 旁路独立过程链（bypass_span）──────────────────────


async def test_bypass_span_records_independent_trace(db):
    """bypass_span 内 LLM 调用产出独立 trace（trace_type 正确、聚合口径）"""
    async with trace_collector.bypass_span(
        conversation_id=2, message_id=None, user_id=42, model_id="gpt-x", trace_type="summary"
    ):
        current = trace_collector.current()
        assert current is not None
        bypass_trace_id = current.trace_id
        call = trace_collector.begin_llm_call("gpt-x", [], None, None)
        assert call is not None
        call.observe_chunk(_chunk("done", usage={"prompt_tokens": 7, "completion_tokens": 3}))
        await call.finish(completed=True)
    await _drain()
    assert trace_collector.current() is None  # 退出后 ContextVar 清理
    trace = await ai_trace_repository.get_by_trace_id(db, bypass_trace_id)
    assert trace is not None
    assert trace.trace_type == "summary"
    assert trace.message_id is None
    assert trace.llm_call_count == 1
    assert trace.prompt_tokens == 7
    assert trace.completion_tokens == 3
    assert trace.total_tokens == 10


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
        current = trace_collector.current()
        assert current is not None
        assert current.trace_type == "memory_extraction"
    assert trace_collector.current() is main


async def test_bypass_span_failure_finalizes_then_reraises(db):
    """异常路径：落失败态后原样抛出（调用方兜底仍生效），ContextVar 已清理"""
    trace_id = None

    async def _run():
        nonlocal trace_id
        async with trace_collector.bypass_span(
            conversation_id=1, message_id=None, user_id=None, model_id="m", trace_type="suggestion"
        ):
            current = trace_collector.current()
            assert current is not None
            trace_id = current.trace_id
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await _run()
    assert trace_collector.current() is None
    await _drain()
    assert trace_id is not None
    trace = await ai_trace_repository.get_by_trace_id(db, trace_id)
    assert trace is not None
    assert trace.trace_type == "suggestion"
    assert trace.status == trace_collector.TRACE_STATUS_FAILED
    assert trace.error_type == "RuntimeError"


async def test_main_trace_defaults_to_conversation_type(db):
    collector = _collector()
    await collector.settle(status=trace_collector.TRACE_STATUS_SUCCESS)
    await _drain()
    trace = await ai_trace_repository.get_by_trace_id(db, collector.trace_id)
    assert trace is not None
    assert trace.trace_type == "conversation"


# ── 消息级聚合落盘（§2.4）─────────────────────────────


async def test_settle_writes_trace_with_billing_usage(db):
    collector = _collector()
    collector.record_context(
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
        injected_memories=[],
        summary=None,
    )
    await collector.settle(
        status=trace_collector.TRACE_STATUS_SUCCESS,
        usage={"input_tokens": 100, "output_tokens": 50, "cached_input_tokens": 20},
        step_count=3,
        actual_model="fallback-model",
    )
    await _drain()
    trace = await ai_trace_repository.get_by_trace_id(db, collector.trace_id)
    assert trace is not None
    assert trace.status == 1
    assert trace.model == "fallback-model"  # 实际路由归因（降级场景）
    assert trace.total_tokens == 150
    assert trace.cached_tokens == 20
    assert trace.step_count == 3
    assert trace.llm_call_count == 0
    assert trace.first_token_ms is None
    context_snapshot = trace.context_snapshot
    assert context_snapshot is not None
    assert {i["type"] for i in context_snapshot["items"]} == {"system", "history"}


async def test_settle_idempotent_by_trace_id(db):
    collector = _collector()
    await collector.settle(status=trace_collector.TRACE_STATUS_SUCCESS, step_count=1)
    await collector.settle(
        status=trace_collector.TRACE_STATUS_FAILED, error_type="x"
    )  # 重复结算跳过
    await _drain()
    traces, total = await ai_trace_repository.list_traces(db, conversation_id=1)
    assert total == 1
    assert traces[0].status == 1


async def test_settle_fallback_to_llm_call_aggregation(db):
    """计费 usage 缺失时回退 LLM 调用聚合口径"""
    collector = _collector()
    call = collector.begin_llm_call("gpt-x", [], None, None)
    call.observe_chunk(_chunk("done", usage={"prompt_tokens": 7, "completion_tokens": 3}))
    await call.finish(completed=True)
    await _drain()
    await collector.settle(status=trace_collector.TRACE_STATUS_SUCCESS)
    await _drain()
    trace = await ai_trace_repository.get_by_trace_id(db, collector.trace_id)
    assert trace is not None
    assert trace.llm_call_count == 1
    assert trace.prompt_tokens == 7
    assert trace.completion_tokens == 3
    assert trace.total_tokens == 10


# ── 异常映射与 finalize 辅助 ──────────────────────────


def test_error_type_of():
    assert error_type_of(BusinessException(ResultCode.AI_LLM_CALL_FAILED, "x")) == "A0600"
    assert error_type_of(TimeoutError()) == "TimeoutError"


async def test_finalize_unsettled_writes_failure(db):
    _collector()
    await trace_collector.finalize_unsettled(status=trace_collector.TRACE_STATUS_INTERRUPTED)
    await _drain()
    traces, total = await ai_trace_repository.list_traces(db, conversation_id=1)
    assert total == 1
    assert traces[0].status == 3


async def test_finalize_unsettled_writes_error_detail(db):
    _collector()
    detail = {"message": "boom", "stack": "Traceback ..."}
    await trace_collector.finalize_unsettled(
        status=trace_collector.TRACE_STATUS_FAILED,
        error_type="A0600",
        error_detail=detail,
    )
    await _drain()
    traces, total = await ai_trace_repository.list_traces(db, conversation_id=1)
    assert total == 1
    assert traces[0].error_detail == detail


async def test_record_event_writes_context_events(db):
    collector = _collector()
    collector.record_event(event="guardrail", rule="prompt_injection", detail="命中关键词")
    collector.record_event(event="plan", phase="plan", plan_summary='{"tasks":[]}')
    await collector.settle(status=trace_collector.TRACE_STATUS_SUCCESS)
    await _drain()
    trace = await ai_trace_repository.get_by_trace_id(db, collector.trace_id)
    assert trace is not None
    context_snapshot = trace.context_snapshot
    assert context_snapshot is not None
    assert {
        "event": "guardrail",
        "rule": "prompt_injection",
        "detail": "命中关键词",
    } in context_snapshot["events"]
    assert {
        "event": "plan",
        "phase": "plan",
        "plan_summary": '{"tasks":[]}',
    } in context_snapshot["events"]


async def test_record_event_skipped_after_settle():
    collector = _collector()
    collector._settled = True  # 已结算（不触发落库，无 db fixture 不得走真实引擎）
    collector.record_event(event="guardrail", rule="x", detail="y")  # 已结算不写入
    assert collector.context_events == []


async def test_finalize_helpers_noop_without_collector():
    await trace_collector.finalize_unsettled(status=2)  # 无采集器不抛错
    await trace_collector.finalize_success()


# ── 落盘后台化与 fail-open ────────────────────────────


async def test_llm_call_finish_writes_in_background(db):
    """finish 不阻塞：返回时写库任务已派发但未执行，drain 后落库可见"""
    collector = _collector()
    call = collector.begin_llm_call("gpt-x", [], None, None)
    await call.finish(completed=True)
    assert trace_collector._pending_tasks  # 任务已登记（持有引用防 GC）
    assert await ai_llm_call_repository.list_by_trace(db, collector.trace_id) == []

    await _drain()
    assert len(await ai_llm_call_repository.list_by_trace(db, collector.trace_id)) == 1
    assert not trace_collector._pending_tasks  # done 回调已回收引用


async def test_settle_writes_in_background(db):
    collector = _collector()
    await collector.settle(status=trace_collector.TRACE_STATUS_SUCCESS)
    assert await ai_trace_repository.get_by_trace_id(db, collector.trace_id) is None

    await _drain()
    assert await ai_trace_repository.get_by_trace_id(db, collector.trace_id) is not None


async def test_trace_write_failure_does_not_raise(db, monkeypatch):
    """过程链落盘失败仅告警，不得冒泡到推理收尾链路"""

    async def _boom(db, values):
        raise RuntimeError("db down")

    monkeypatch.setattr(ai_trace_repository, "insert_idempotent", _boom)
    collector = _collector()
    await collector.settle(status=trace_collector.TRACE_STATUS_SUCCESS)
    await _drain()
    assert collector.settled


async def test_settle_spawn_failure_does_not_raise(monkeypatch):
    """落盘派发本身失败（无事件循环等）时同样 fail-open"""

    def _boom(*args, **kwargs):
        raise RuntimeError("no running loop")

    monkeypatch.setattr(trace_collector, "_spawn_persist", _boom)
    collector = _collector()
    await collector.settle(status=trace_collector.TRACE_STATUS_SUCCESS)
    assert collector.settled


async def test_finish_idempotent_under_concurrency(db):
    """并发重复 finish：_finished 同步置位，只落一条明细"""
    collector = _collector()
    call = collector.begin_llm_call("gpt-x", [], None, None)
    await asyncio.gather(
        call.finish(completed=True), call.finish(completed=False, error_type="5xx")
    )
    await _drain()
    calls = await ai_llm_call_repository.list_by_trace(db, collector.trace_id)
    assert len(calls) == 1
    assert calls[0].status == 1
