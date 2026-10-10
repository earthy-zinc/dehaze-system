"""工具调用参数流式聚合回归

背景（真实缺陷）：协议客户端产出的 `tool_call_delta` 未携带 `tool_call_id`，
`DehazeChatModel._astream` 按 id 匹配暂存缓冲区，匹配失败即静默丢弃参数分片，
导致**所有工具调用参数恒为空 `{}`**（openai_compat / anthropic 两条链路均受影响）。
本文件锁定：参数分片必须带 id，且最终聚合出的工具调用参数非空。
"""

import re
from types import SimpleNamespace

import httpx
import respx
from langchain_core.messages import HumanMessage

from app.infrastructure.llm.client import dehaze_chat_model as m
from app.infrastructure.llm.client.anthropic_client import AnthropicClient
from app.infrastructure.llm.client.dehaze_chat_model import DehazeChatModel
from app.infrastructure.llm.client.openai_compat_client import OpenAiCompatClient
from tests.stubs.fakes import NullDBSession

CHAT_URL_RE = re.compile(r"https://api\.example\.com/v1/chat/completions")
MESSAGES_URL_RE = re.compile(r"https://api\.example\.com/v1/messages")


def _model(model_id="qwen", pk=1):
    return SimpleNamespace(
        id=pk,
        model_id=model_id,
        provider_id=1,
        max_output_tokens=2048,
        supports_prompt_cache=0,
        prompt_cache_prefix_len=0,
        extra_request_params=None,
        status=1,
    )


def _provider(protocol="openai_compat", auth="bearer"):
    return SimpleNamespace(
        id=1,
        protocol_type=protocol,
        auth_type=auth,
        api_base_url="https://api.example.com/v1",
        default_headers={},
        status=1,
    )


def _sse(lines):
    return "\n".join(lines).encode("utf-8")


async def test_openai_tool_call_delta_carries_id():
    """openai_compat：参数分片携带 tool_call_id，且可拼接为完整 JSON"""
    lines = [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
        '"function":{"name":"knowledge_base_search","arguments":""}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
        '"function":{"arguments":"{\\"query\\":"}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
        '"function":{"arguments":"\\"去雾\\"}"}}]}}]}',
        "data: [DONE]",
    ]
    with respx.mock(assert_all_mocked=True) as router:
        router.post(CHAT_URL_RE).mock(
            return_value=httpx.Response(
                200, content=_sse(lines), headers={"content-type": "text/event-stream"}
            )
        )
        transport = httpx.AsyncClient()
        chunks = [
            c
            async for c in OpenAiCompatClient(transport).stream_chat(
                _provider(), "sk-x", _model(), [], None, None, None, None, 0.7
            )
        ]

    deltas = [c for c in chunks if c.type == "tool_call_delta"]
    assert [c.tool_call_id for c in deltas] == ["call_1", "call_1"]
    assert "".join(c.content for c in deltas) == '{"query":"去雾"}'


async def test_anthropic_tool_call_delta_carries_id():
    """anthropic：input_json_delta 分片同样携带 tool_call_id"""
    lines = [
        'data: {"type":"content_block_start","index":0,"content_block":'
        '{"type":"tool_use","id":"toolu_1","name":"knowledge_base_search","input":{}}}',
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"input_json_delta","partial_json":"{\\"query\\":"}}',
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"input_json_delta","partial_json":"\\"去雾\\"}"}}',
        'data: {"type":"content_block_stop","index":0}',
    ]
    with respx.mock(assert_all_mocked=True) as router:
        router.post(MESSAGES_URL_RE).mock(
            return_value=httpx.Response(
                200, content=_sse(lines), headers={"content-type": "text/event-stream"}
            )
        )
        transport = httpx.AsyncClient()
        chunks = [
            c
            async for c in AnthropicClient(transport).stream_chat(
                _provider(protocol="anthropic", auth="x-api-key"),
                "sk-x",
                _model("claude-3-5-sonnet"),
                [],
                None,
                None,
                None,
                None,
            )
        ]

    deltas = [c for c in chunks if c.type == "tool_call_delta"]
    assert [c.tool_call_id for c in deltas] == ["toolu_1", "toolu_1"]
    assert "".join(c.content for c in deltas) == '{"query":"去雾"}'


async def test_dehaze_chat_model_aggregates_tool_args(monkeypatch):
    """端到端：参数分片按 id 聚合进最终工具调用（修复前 args 恒为 {}）"""
    monkeypatch.setattr(m, "get_db_session", lambda: NullDBSession())

    async def _stream_chat(db, model_id, messages, **kwargs):
        yield SimpleNamespace(
            type="tool_call_start",
            tool_call_id="call_1",
            tool_call_name="knowledge_base_search",
            content="",
        )
        yield SimpleNamespace(
            type="tool_call_delta", tool_call_id="call_1", tool_call_name="", content='{"query":'
        )
        yield SimpleNamespace(
            type="tool_call_delta", tool_call_id="call_1", tool_call_name="", content='"去雾"}'
        )
        yield SimpleNamespace(
            type="tool_call_complete",
            tool_call_id="call_1",
            tool_call_name="knowledge_base_search",
            content='{"query":"去雾"}',
        )
        yield SimpleNamespace(type="done", usage={"input_tokens": 1, "output_tokens": 1})

    monkeypatch.setattr(m.llm_client, "stream_chat", _stream_chat)

    result = await DehazeChatModel(model="qwen").ainvoke([HumanMessage(content="检索")])
    assert result.tool_calls, "应聚合出工具调用"
    assert result.tool_calls[0]["name"] == "knowledge_base_search"
    assert result.tool_calls[0]["args"] == {"query": "去雾"}
