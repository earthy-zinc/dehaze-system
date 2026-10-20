"""协议客户端 wire 级原始报文采集测试（审计级重构 P0）

验证 openai_compat / anthropic 客户端在请求构建处与流聚合处，
经 trace_collector.record_wire_* 通道上报 wire 原文：
raw_request = 实际发送的完整请求体；raw_response = 流式聚合的等价非流式结构。
"""

import json
import re
from types import SimpleNamespace

import httpx
import respx

from app.infrastructure.llm.client.anthropic_client import AnthropicClient
from app.infrastructure.llm.client.openai_compat_client import OpenAiCompatClient
from app.service.ai.service import trace_collector

CHAT_URL_RE = re.compile(r"https://api\.example\.com/v1/chat/completions")
MESSAGES_URL_RE = re.compile(r"https://api\.example\.com/v1/messages")


def _sse(lines):
    """将 SSE 行序列编码为 httpx 可消费的字节流（aiter_lines 按 \\n 切行）。"""
    return "\n".join(lines).encode("utf-8")


_SSE_HEADERS = {"content-type": "text/event-stream"}


def _make_model(model_id="gpt-x"):
    return SimpleNamespace(
        id=1,
        model_id=model_id,
        provider_id=1,
        max_output_tokens=2048,
        supports_prompt_cache=0,
        prompt_cache_prefix_len=0,
        extra_request_params=None,
        status=1,
    )


def _make_provider(protocol="openai_compat"):
    return SimpleNamespace(
        id=1,
        provider_code="openai",
        protocol_type=protocol,
        auth_type="bearer",
        api_base_url="https://api.example.com/v1",
        default_headers={},
        status=1,
    )


def _begin_wire_call(model_id: str):
    """开启采集器并挂载 wire 上报通道，返回 (call, reset)"""
    collector = trace_collector.start(
        conversation_id=1, message_id=None, user_id=None, agent_code=None, model_id=model_id
    )
    call = collector.begin_llm_call(model_id, [], None, None)
    token = trace_collector.wire_record.set(call)

    def _reset():
        trace_collector.wire_record.reset(token)
        trace_collector._current_collector.set(None)

    return call, _reset


async def test_openai_wire_request_is_actual_payload():
    captured = {}

    def _handler(request):
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content=_sse(['data: {"choices":[{"delta":{"content":"ok"}}]}', "data: [DONE]"]),
            headers=_SSE_HEADERS,
        )

    call, reset = _begin_wire_call("gpt-x")
    try:
        with respx.mock(assert_all_mocked=True) as router:
            router.post(CHAT_URL_RE).mock(side_effect=_handler)
            client = httpx.AsyncClient()
            messages = [{"role": "user", "content": "hi"}]
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "搜索",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                    },
                }
            ]
            _ = [
                c
                async for c in OpenAiCompatClient(client).stream_chat(
                    _make_provider(),
                    "sk-x",
                    _make_model(),
                    messages,
                    "sys",
                    1024,
                    tools,
                    "auto",
                    0.7,
                )
            ]
    finally:
        reset()

    # raw_request 与实际发送请求体逐字段一致（含 tools 完整 JSON Schema/tool_choice/temperature）
    payload = captured["payload"]
    assert call._raw_request == payload
    assert payload["tools"][0]["function"]["parameters"] == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
    }
    assert payload["tool_choice"] == "auto"
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 1024
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


async def test_openai_wire_response_aggregates_nonstream_structure():
    lines = [
        'data: {"id":"chatcmpl-1","model":"gpt-x","choices":['
        '{"index":0,"delta":{"role":"assistant","content":"查询前"}}]}',
        'data: {"id":"chatcmpl-1","model":"gpt-x","choices":['
        '{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_9",'
        '"function":{"name":"search","arguments":"{\\"q\\":"}}]}}]}',
        'data: {"id":"chatcmpl-1","model":"gpt-x","choices":['
        '{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}',
        'data: {"id":"chatcmpl-1","model":"gpt-x","choices":['
        '{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
        'data: {"id":"chatcmpl-1","model":"gpt-x","choices":[],"usage":'
        '{"prompt_tokens":478,"completion_tokens":24,"total_tokens":502,'
        '"prompt_tokens_details":{"cached_tokens":7}}}',
        "data: [DONE]",
    ]
    call, reset = _begin_wire_call("gpt-x")
    try:
        with respx.mock(assert_all_mocked=True) as router:
            router.post(CHAT_URL_RE).mock(
                return_value=httpx.Response(200, content=_sse(lines), headers=_SSE_HEADERS)
            )
            client = httpx.AsyncClient()
            _ = [
                c
                async for c in OpenAiCompatClient(client).stream_chat(
                    _make_provider(), "sk-x", _make_model(), [], None, None, None, None
                )
            ]
    finally:
        reset()

    raw = call._raw_response
    assert raw is not None
    assert raw["id"] == "chatcmpl-1"
    assert raw["model"] == "gpt-x"
    assert raw["choices"][0]["finish_reason"] == "tool_calls"
    message = raw["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert message["content"] == "查询前"
    # tool_calls arguments 为流式分块拼接后的原文
    assert message["tool_calls"] == [
        {
            "id": "call_9",
            "type": "function",
            "function": {"name": "search", "arguments": '{"q":1}'},
        }
    ]
    # usage 为 provider 原始字段（含 cached）
    assert raw["usage"]["prompt_tokens"] == 478
    assert raw["usage"]["prompt_tokens_details"]["cached_tokens"] == 7


async def test_openai_wire_noop_without_wire_context():
    """无 wire 上下文（无采集器）时采集静默跳过，不影响流解析"""
    lines = ['data: {"choices":[{"delta":{"content":"ok"}}]}', "data: [DONE]"]
    with respx.mock(assert_all_mocked=True) as router:
        router.post(CHAT_URL_RE).mock(
            return_value=httpx.Response(200, content=_sse(lines), headers=_SSE_HEADERS)
        )
        client = httpx.AsyncClient()
        chunks = [
            c
            async for c in OpenAiCompatClient(client).stream_chat(
                _make_provider(), "sk-x", _make_model(), [], None, None, None, None
            )
        ]
    assert [c.type for c in chunks] == ["text_delta"]
    assert trace_collector.wire_record.get() is None


async def test_anthropic_wire_request_is_actual_payload():
    captured = {}

    def _handler(request):
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content=_sse(['data: {"type":"message_start","message":{"usage":{}}}', "data: [DONE]"]),
            headers=_SSE_HEADERS,
        )

    call, reset = _begin_wire_call("claude-x")
    try:
        with respx.mock(assert_all_mocked=True) as router:
            router.post(MESSAGES_URL_RE).mock(side_effect=_handler)
            client = httpx.AsyncClient()
            tools = [{"type": "function", "function": {"name": "t1", "description": "d"}}]
            _ = [
                c
                async for c in AnthropicClient(client).stream_chat(
                    _make_provider(protocol="anthropic"),
                    "sk-x",
                    _make_model("claude-x"),
                    [],
                    "SYSTEM",
                    1024,
                    tools,
                    "auto",
                    0.7,
                )
            ]
    finally:
        reset()

    payload = captured["payload"]
    assert call._raw_request == payload
    # anthropic 原生格式：system 顶层、tools 为 input_schema 形态
    assert payload["system"] == "SYSTEM"
    assert payload["tools"] == [
        {"name": "t1", "description": "d", "input_schema": {"type": "object", "properties": {}}}
    ]
    assert payload["tool_choice"] == {"type": "auto"}
    assert payload["max_tokens"] == 1024


async def test_anthropic_wire_response_aggregates_nonstream_structure():
    lines = [
        'data: {"type":"message_start","message":{"id":"msg_1","model":"claude-x","usage":'
        '{"input_tokens":10,"cache_read_input_tokens":4}}}',
        'data: {"type":"content_block_start","index":0,"content_block":'
        '{"type":"tool_use","id":"toolu_1","name":"search"}}',
        'data: {"type":"content_block_delta","index":0,"delta":'
        '{"type":"input_json_delta","partial_json":"{\\"q\\":"}}',
        'data: {"type":"content_block_delta","index":0,"delta":'
        '{"type":"input_json_delta","partial_json":"1}"}}',
        'data: {"type":"content_block_stop","index":0}',
        'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
        'data: {"type":"content_block_delta","index":1,"delta":'
        '{"type":"text_delta","text":"好的"}}',
        'data: {"type":"content_block_stop","index":1}',
        'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},'
        '"usage":{"output_tokens":5}}',
    ]
    call, reset = _begin_wire_call("claude-x")
    try:
        with respx.mock(assert_all_mocked=True) as router:
            router.post(MESSAGES_URL_RE).mock(
                return_value=httpx.Response(200, content=_sse(lines), headers=_SSE_HEADERS)
            )
            client = httpx.AsyncClient()
            _ = [
                c
                async for c in AnthropicClient(client).stream_chat(
                    _make_provider(protocol="anthropic"),
                    "sk-x",
                    _make_model("claude-x"),
                    [],
                    None,
                    None,
                    None,
                    None,
                )
            ]
    finally:
        reset()

    raw = call._raw_response
    assert raw is not None
    assert raw["id"] == "msg_1"
    assert raw["model"] == "claude-x"
    # anthropic stop_reason 映射为统一 finish_reason 口径
    assert raw["choices"][0]["finish_reason"] == "tool_calls"
    message = raw["choices"][0]["message"]
    assert message["content"] == "好的"
    assert message["tool_calls"] == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "search", "arguments": '{"q":1}'},
        }
    ]
    # usage 为 provider 原始字段（含缓存命中 cache_read_input_tokens）
    assert raw["usage"] == {"input_tokens": 10, "cache_read_input_tokens": 4, "output_tokens": 5}
