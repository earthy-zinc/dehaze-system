from types import SimpleNamespace
from unittest.mock import Mock

from app.service.ai.llm_client import LlmClient
from tests.stubs import FakeStreamResponse


def _make_model(model_id="deepseek-r1", pk=1):
    return SimpleNamespace(
        id=pk,
        model_id=model_id,
        provider_id=1,
        max_output_tokens=2048,
        supports_prompt_cache=0,
        prompt_cache_prefix_len=0,
        status=1,
    )


def _make_provider(
    protocol="openai_compat", auth="bearer", api_base_url="https://api.example.com/v1"
):
    return SimpleNamespace(
        id=1,
        protocol_type=protocol,
        auth_type=auth,
        api_base_url=api_base_url,
        default_headers={},
        status=1,
    )


def _make_client() -> tuple[LlmClient, Mock]:
    client = LlmClient.__new__(LlmClient)
    client._client = Mock()
    return client, client._client


class _FakeStream:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        return False


def _stream_lines(lines):
    return _FakeStream(FakeStreamResponse(lines=lines))


async def test_openai_reasoning_content_yields_thinking_delta():
    client, transport = _make_client()
    lines = [
        'data: {"choices":[{"delta":{"reasoning_content":"让我分析"}}]}',
        'data: {"choices":[{"delta":{"reasoning_content":"一下参数"}}]}',
        'data: {"choices":[{"delta":{"content":"最终答案"}}]}',
        "data: [DONE]",
    ]
    transport.stream = Mock(side_effect=[_stream_lines(lines)])
    chunks = [
        c
        async for c in client._stream_openai(
            _make_provider(),
            "sk-x",
            _make_model(),
            [],
            None,
            0.7,
            None,
            None,
            None,
        )
    ]
    assert [c.type for c in chunks] == ["thinking_delta", "thinking_delta", "text_delta"]
    assert chunks[0].content == "让我分析"
    assert chunks[1].content == "一下参数"
    assert chunks[2].content == "最终答案"


async def test_anthropic_thinking_block_streams_and_discards_signature():
    client, transport = _make_client()
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
    transport.stream = Mock(side_effect=[_stream_lines(lines)])
    chunks = [
        c
        async for c in client._stream_anthropic(
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
    client, transport = _make_client()
    lines = [
        'data: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"thinking","signature":"SIG"},"signature":"SIG"}',
        'data: {"type":"content_block_stop","index":0}',
        'data: {"type":"message_delta","usage":{"output_tokens":5}}',
    ]
    transport.stream = Mock(side_effect=[_stream_lines(lines)])
    chunks = [
        c
        async for c in client._stream_anthropic(
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
