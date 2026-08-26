import json
import re
from types import SimpleNamespace

import httpx
import pytest
import respx

from app.infrastructure.llm.client.anthropic_client import AnthropicClient
from app.infrastructure.llm.client.openai_compat_client import OpenAiCompatClient

# 被测客户端按 provider.api_base_url 动态拼接：OpenAI 兼容 = {base}/chat/completions，
# Anthropic = {base}/messages。用正则路由匹配屏蔽变量基址。
CHAT_URL_RE = re.compile(r"https://api\.example\.com/v1/chat/completions")
MESSAGES_URL_RE = re.compile(r"https://api\.example\.com/v1/messages")


def _make_model(model_id="deepseek-r1", pk=1, extra_request_params=None):
    return SimpleNamespace(
        id=pk,
        model_id=model_id,
        provider_id=1,
        max_output_tokens=2048,
        supports_prompt_cache=0,
        prompt_cache_prefix_len=0,
        extra_request_params=extra_request_params,
        status=1,
    )


def _make_provider(protocol="openai_compat", auth="bearer", api_base_url="https://api.example.com/v1"):
    return SimpleNamespace(
        id=1,
        protocol_type=protocol,
        auth_type=auth,
        api_base_url=api_base_url,
        default_headers={},
        status=1,
    )


def _sse(lines):
    """将 SSE 行序列编码为 httpx 可消费的字节流（aiter_lines 按 \\n 切行）。"""
    return "\n".join(lines).encode("utf-8")


async def test_openai_reasoning_content_yields_thinking_delta():
    lines = [
        'data: {"choices":[{"delta":{"reasoning_content":"让我分析"}}]}',
        'data: {"choices":[{"delta":{"reasoning_content":"一下参数"}}]}',
        'data: {"choices":[{"delta":{"content":"最终答案"}}]}',
        "data: [DONE]",
    ]
    with respx.mock(assert_all_mocked=True) as router:
        router.post(CHAT_URL_RE).mock(
            return_value=httpx.Response(200, content=_sse(lines), headers={"content-type": "text/event-stream"})
        )
        transport = httpx.AsyncClient()
        chunks = [
            c
            async for c in OpenAiCompatClient(transport).stream_chat(
                _make_provider(),
                "sk-x",
                _make_model(),
                [],
                None,
                None,
                None,
                None,
                0.7,
            )
        ]
    assert [c.type for c in chunks] == ["thinking_delta", "thinking_delta", "text_delta"]
    assert chunks[0].content == "让我分析"
    assert chunks[1].content == "一下参数"
    assert chunks[2].content == "最终答案"


async def test_extra_request_params_merged_and_core_keys_protected():
    """模型配置的厂商私有参数合并进请求体，核心键（stream/model/messages 等）不可覆盖"""
    lines = ['data: {"choices":[{"delta":{"content":"ok"}}]}', "data: [DONE]"]
    captured: dict = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200, content=_sse(lines), headers={"content-type": "text/event-stream"}
        )

    with respx.mock(assert_all_mocked=True) as router:
        router.post(CHAT_URL_RE).mock(side_effect=handler)
        transport = httpx.AsyncClient()
        chunks = [
            c
            async for c in OpenAiCompatClient(transport).stream_chat(
                _make_provider(),
                "sk-x",
                _make_model(
                    "qwen3",
                    extra_request_params={"enable_thinking": False, "stream": "hacked"},
                ),
                [],
                None,
                None,
                None,
                None,
                0.7,
            )
        ]
    assert [c.type for c in chunks] == ["text_delta"]
    assert captured["enable_thinking"] is False
    assert captured["stream"] is True  # 核心键不可被配置覆盖
    assert captured["model"] == "qwen3"
    assert captured["max_tokens"] == 2048


async def test_anthropic_thinking_block_streams_and_discards_signature():
    lines = [
        'data: {"type":"content_block_start","index":0,"content_block":'
        '{"type":"thinking","thinking":"思考中","signature":"SIG123"}}',
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"thinking_delta","thinking":"继续想"}}',
        'data: {"type":"content_block_stop","index":0}',
        'data: {"type":"content_block_start","index":1,'
        '"content_block":{"type":"text","text":""}}',
        'data: {"type":"content_block_delta","index":1,'
        '"delta":{"type":"text_delta","text":"正文"}}',
        'data: {"type":"content_block_stop","index":1}',
        'data: {"type":"message_delta","usage":{"output_tokens":10}}',
    ]
    with respx.mock(assert_all_mocked=True) as router:
        router.post(MESSAGES_URL_RE).mock(
            return_value=httpx.Response(200, content=_sse(lines), headers={"content-type": "text/event-stream"})
        )
        transport = httpx.AsyncClient()
        chunks = [
            c
            async for c in AnthropicClient(transport).stream_chat(
                _make_provider(protocol="anthropic", auth="x-api-key"),
                "sk-x",
                _make_model("claude-3-5-sonnet"),
                [],
                None,
                None,
                None,
                None,
            )
        ]
    assert [c.type for c in chunks] == ["thinking_delta", "thinking_delta", "text_delta", "done"]
    assert chunks[0].content == "思考中"
    assert chunks[1].content == "继续想"
    assert chunks[2].content == "正文"
    assert all("SIG123" not in c.content for c in chunks)


async def test_anthropic_thinking_start_without_text_still_streams():
    lines = [
        'data: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"thinking","signature":"SIG"},"signature":"SIG"}',
        'data: {"type":"content_block_stop","index":0}',
        'data: {"type":"message_delta","usage":{"output_tokens":5}}',
    ]
    with respx.mock(assert_all_mocked=True) as router:
        router.post(MESSAGES_URL_RE).mock(
            return_value=httpx.Response(200, content=_sse(lines), headers={"content-type": "text/event-stream"})
        )
        transport = httpx.AsyncClient()
        chunks = [
            c
            async for c in AnthropicClient(transport).stream_chat(
                _make_provider(protocol="anthropic", auth="x-api-key"),
                "sk-x",
                _make_model("claude-3-5-sonnet"),
                [],
                None,
                None,
                None,
                None,
            )
        ]
    assert [c.type for c in chunks] == ["done"]
