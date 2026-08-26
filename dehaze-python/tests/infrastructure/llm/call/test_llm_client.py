import json
import re
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
import respx

from app.core.exceptions import BusinessException
from app.infrastructure.llm.call.llm_client import LlmClient

# 被测代码按 provider.api_base_url 动态拼接 URL：{base}/chat/completions（OpenAI 兼容）
# 或 {base}/messages（Anthropic）。测试用正则路由匹配，屏蔽变量基址差异。
CHAT_URL_RE = re.compile(r"https://api\.example\.com/v1/chat/completions")
MESSAGES_URL_RE = re.compile(r"https://api\.example\.com/v1/messages")


def _make_model(model_id="gpt-4o", pk=1, provider_id=1, **overrides):
    base = {
        "id": pk,
        "model_id": model_id,
        "provider_id": provider_id,
        "max_output_tokens": 2048,
        "supports_prompt_cache": 0,
        "prompt_cache_prefix_len": 0,
        "extra_request_params": None,
        "status": 1,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_provider(
    provider_id=1,
    protocol="openai_compat",
    auth="bearer",
    api_base_url="https://api.example.com/v1",
    default_headers=None,
    provider_code="openai",
):
    return SimpleNamespace(
        id=provider_id,
        provider_code=provider_code,
        protocol_type=protocol,
        auth_type=auth,
        api_base_url=api_base_url,
        default_headers=default_headers or {},
        status=1,
    )


def _make_key(key_id, provider_id=1, priority=0, weight=1, daily_quota=None):
    return SimpleNamespace(
        id=key_id,
        provider_id=provider_id,
        priority=priority,
        weight=weight,
        daily_quota=daily_quota,
        key_cipher=f"cipher-{key_id}",
    )


def _make_client() -> LlmClient:
    """构造真实 httpx 连接池的 LlmClient，由 respx 在传输层拦截外部请求。"""
    client = LlmClient.__new__(LlmClient)
    client._client = httpx.AsyncClient()
    client._redis = None
    return client


@contextmanager
def _patch_route_io(model, provider):
    provider_mock = (
        provider if isinstance(provider, AsyncMock) else AsyncMock(return_value=provider)
    )
    with (
        patch(
            "app.infrastructure.llm.call.llm_client.ai_model_repository.get_by_id",
            AsyncMock(return_value=model),
        ),
        patch(
            "app.infrastructure.llm.call.llm_client.ai_provider_repository.get_by_id",
            provider_mock,
        ),
        patch("app.infrastructure.llm.call.llm_client.decrypt", side_effect=lambda c: f"sk-{c}"),
    ):
        yield


def _sse(lines):
    """将 SSE 行序列编码为 httpx 可消费的字节流（aiter_lines 按 \\n 切行）。"""
    return "\n".join(lines).encode("utf-8")


def _ok_lines(text="hello"):
    return [
        f'data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}',
        "data: [DONE]",
    ]


class _Services:
    def __init__(self, mocks, patches):
        self._patches = patches
        self.get_call_routes = mocks["get_call_routes"]
        self.mark_call_failed = mocks["mark_call_failed"]
        self.mark_call_success = mocks["mark_call_success"]
        self.get_status = mocks["get_status"]
        self.record_call = mocks["record_call"]
        self.list_usable_keys = mocks["list_usable_keys"]

    def stop(self):
        for p in self._patches.values():
            p.stop()


def _patch_cross_services(get_call_routes, usable_keys=None, redis_client=None):
    from app.infrastructure.provider.model_registry import model_registry
    from app.infrastructure.provider.provider_health_service import provider_health_service
    from app.infrastructure.provider.provider_key_selector import provider_key_selector

    mocks = {
        "get_call_routes": AsyncMock(return_value=get_call_routes),
        "mark_call_failed": AsyncMock(return_value=None),
        "mark_call_success": AsyncMock(return_value=None),
        "get_status": AsyncMock(return_value="healthy"),
        "record_call": AsyncMock(return_value=None),
        "list_usable_keys": AsyncMock(return_value=usable_keys or []),
    }
    patches = {
        "get_call_routes": patch.object(
            model_registry, "get_call_routes", new=mocks["get_call_routes"]
        ),
        "mark_call_failed": patch.object(
            provider_key_selector, "mark_call_failed", new=mocks["mark_call_failed"]
        ),
        "mark_call_success": patch.object(
            provider_key_selector, "mark_call_success", new=mocks["mark_call_success"]
        ),
        "get_status": patch.object(provider_health_service, "get_status", new=mocks["get_status"]),
        "record_call": patch.object(provider_health_service, "record_call", new=mocks["record_call"]),
        "list_usable_keys": patch.object(
            provider_key_selector, "list_usable_keys", new=mocks["list_usable_keys"]
        ),
    }
    if redis_client is not None:
        patches["get_redis_client"] = patch(
            "app.infrastructure.llm.call.llm_client.get_redis_client",
            new=AsyncMock(return_value=redis_client),
        )
    for p in patches.values():
        p.start()
    return _Services(mocks, patches)


async def test_429_switch_key(mock_redis):
    route = {"model_pk": 1, "model_id": "gpt-4o", "provider_id": 1, "model_config": {}}
    svc = _patch_cross_services(
        [route], usable_keys=[_make_key(1), _make_key(2)], redis_client=mock_redis
    )
    model = _make_model()
    provider = _make_provider()
    try:
        with (
            _patch_route_io(model, provider),
            respx.mock(assert_all_mocked=True) as router,
        ):
            router.post(CHAT_URL_RE).mock(
                side_effect=[
                    httpx.Response(429, content=b""),
                    httpx.Response(200, content=_sse(_ok_lines()), headers=_sse_headers()),
                ]
            )
            client = _make_client()
            meta = {}
            chunks = [
                c
                async for c in client.stream_chat(
                    db=None,
                    model_id="gpt-4o",
                    messages=[],
                    system_prompt=None,
                    on_route_result=meta.update,
                )
            ]
    finally:
        svc.stop()

    svc.mark_call_failed.assert_called_once_with(mock_redis, 1, "429")
    svc.mark_call_success.assert_called_once()
    assert meta["key_id"] == 2
    assert [c.type for c in chunks] == ["text_delta"]


async def test_key_exhausted_switch_provider(mock_redis):
    routes = [
        {"model_pk": 1, "model_id": "gpt-4o", "provider_id": 1, "model_config": {}},
        {"model_pk": 1, "model_id": "gpt-4o", "provider_id": 2, "model_config": {}},
    ]
    svc = _patch_cross_services(routes, redis_client=mock_redis)
    svc.list_usable_keys.side_effect = [
        [_make_key(1, provider_id=1)],
        [_make_key(3, provider_id=2)],
    ]
    model = _make_model()
    provider_a = _make_provider(provider_id=1)
    provider_b = _make_provider(provider_id=2)
    try:
        with (
            _patch_route_io(model, AsyncMock(side_effect=[provider_a, provider_b])),
            respx.mock(assert_all_mocked=True) as router,
        ):
            router.post(CHAT_URL_RE).mock(
                side_effect=[
                    httpx.Response(429, content=b""),
                    httpx.Response(200, content=_sse(_ok_lines("ok")), headers=_sse_headers()),
                ]
            )
            client = _make_client()
            meta = {}
            chunks = [
                c
                async for c in client.stream_chat(
                    db=None,
                    model_id="gpt-4o",
                    messages=[],
                    on_route_result=meta.update,
                )
            ]
    finally:
        svc.stop()

    svc.mark_call_failed.assert_called_once_with(mock_redis, 1, "429")
    assert meta["provider_id"] == 2
    assert meta["key_id"] == 3
    assert [c.content for c in chunks] == ["ok"]


async def test_all_fail_raise_business_exception(mock_redis):
    route = {"model_pk": 1, "model_id": "gpt-4o", "provider_id": 1, "model_config": {}}
    svc = _patch_cross_services([route], usable_keys=[_make_key(1)], redis_client=mock_redis)
    model = _make_model()
    provider = _make_provider()
    try:
        with (
            _patch_route_io(model, provider),
            respx.mock(assert_all_mocked=True) as router,
        ):
            router.post(CHAT_URL_RE).mock(return_value=httpx.Response(500, content=b""))
            client = _make_client()
            with pytest.raises(BusinessException) as exc:
                async for _ in client.stream_chat(
                    db=None, model_id="gpt-4o", messages=[]
                ):
                    pass
    finally:
        svc.stop()

    assert "主模型和降级模型均不可用" in str(exc.value.message)


async def test_required_caps_passed(mock_redis):
    route = {"model_pk": 1, "model_id": "gpt-4o", "provider_id": 1, "model_config": {}}
    svc = _patch_cross_services([route], usable_keys=[_make_key(1)], redis_client=mock_redis)
    model = _make_model()
    provider = _make_provider()
    captured_caps = None
    try:
        with (
            _patch_route_io(model, provider),
            respx.mock(assert_all_mocked=True) as router,
        ):
            router.post(CHAT_URL_RE).mock(
                return_value=httpx.Response(200, content=_sse(_ok_lines()), headers=_sse_headers())
            )
            client = _make_client()
            async for _ in client.stream_chat(
                db=None,
                model_id="gpt-4o",
                messages=[],
                tools=[{"type": "function", "function": {"name": "f"}}],
            ):
                pass
            captured_caps = svc.get_call_routes.await_args.args[2]
    finally:
        svc.stop()

    assert captured_caps == {"streaming", "tool_call"}


async def test_anthropic_cache_control_injected(mock_redis):
    route = {"model_pk": 1, "model_id": "claude-3-5-sonnet", "provider_id": 1, "model_config": {}}
    svc = _patch_cross_services([route], usable_keys=[_make_key(1)], redis_client=mock_redis)
    model = _make_model(
        model_id="claude-3-5-sonnet",
        provider_id=1,
        supports_prompt_cache=1,
        prompt_cache_prefix_len=100,
    )
    provider = _make_provider(provider_id=1, protocol="anthropic", auth="x-api-key")
    captured = {}

    def _handler(request):
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content=_sse(
                [
                    'data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}',
                    'data: {"type":"message_delta","delta":{},"usage":{"output_tokens":5}}',
                ]
            ),
            headers={"content-type": "text/event-stream"},
        )

    try:
        with (
            _patch_route_io(model, provider),
            respx.mock(assert_all_mocked=True) as router,
        ):
            router.post(MESSAGES_URL_RE).mock(side_effect=_handler)
            client = _make_client()
            async for _ in client.stream_chat(
                db=None,
                model_id="claude-3-5-sonnet",
                messages=[],
                system_prompt="SYSTEM",
                tools=[
                    {"type": "function", "function": {"name": "t1"}},
                    {"type": "function", "function": {"name": "t2"}},
                ],
            ):
                pass
    finally:
        svc.stop()

    payload = captured["payload"]
    assert payload["system"] == [
        {"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}
    ]
    assert payload["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in payload["tools"][0]


async def test_stream_interrupt_raises_no_switch(mock_redis):
    route = {"model_pk": 1, "model_id": "gpt-4o", "provider_id": 1, "model_config": {}}
    svc = _patch_cross_services(
        [route], usable_keys=[_make_key(1), _make_key(2)], redis_client=mock_redis
    )
    model = _make_model()
    provider = _make_provider()

    async def _broken():
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n'
        raise httpx.ReadError("stream broken")

    try:
        with (
            _patch_route_io(model, provider),
            respx.mock(assert_all_mocked=True) as router,
        ):
            router.post(CHAT_URL_RE).mock(return_value=httpx.Response(200, content=_broken()))
            client = _make_client()
            seen = []
            with pytest.raises(BusinessException) as exc:
                async for c in client.stream_chat(
                    db=None, model_id="gpt-4o", messages=[]
                ):
                    seen.append(c)
    finally:
        svc.stop()

    assert [c.content for c in seen] == ["hi"]
    svc.mark_call_failed.assert_called_once_with(mock_redis, 1, "transport")
    assert "流式响应中断" in str(exc.value.message)


async def test_circuit_open_skips_provider_route(mock_redis):
    routes = [
        {"model_pk": 1, "model_id": "gpt-4o", "provider_id": 1, "model_config": {}},
        {"model_pk": 1, "model_id": "gpt-4o", "provider_id": 2, "model_config": {}},
    ]
    svc = _patch_cross_services(routes, redis_client=mock_redis)
    svc.list_usable_keys.return_value = [_make_key(3)]
    model = _make_model()
    provider_b = _make_provider(provider_id=2)
    try:
        async def _status(redis, provider_id):
            return "open" if provider_id == 1 else "healthy"

        svc.get_status.side_effect = _status
        with (
            _patch_route_io(model, provider_b),
            respx.mock(assert_all_mocked=True) as router,
        ):
            router.post(CHAT_URL_RE).mock(
                return_value=httpx.Response(200, content=_sse(_ok_lines("ok")), headers=_sse_headers())
            )
            client = _make_client()
            meta = {}
            chunks = [
                c
                async for c in client.stream_chat(
                    db=None,
                    model_id="gpt-4o",
                    messages=[],
                    on_route_result=meta.update,
                )
            ]
    finally:
        svc.stop()

    svc.record_call.assert_called_once()
    assert meta["provider_id"] == 2
    assert meta["key_id"] == 3
    assert [c.content for c in chunks] == ["ok"]


def _sse_headers():
    return {"content-type": "text/event-stream"}
