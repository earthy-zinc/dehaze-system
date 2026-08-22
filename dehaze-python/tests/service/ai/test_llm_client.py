from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from app.core.exceptions import BusinessException
from app.service.ai.llm_client import LlmClient
from tests.stubs import FakeStreamResponse


def _make_model(model_id="gpt-4o", pk=1, provider_id=1, **overrides):
    base = {
        "id": pk,
        "model_id": model_id,
        "provider_id": provider_id,
        "max_output_tokens": 2048,
        "supports_prompt_cache": 0,
        "prompt_cache_prefix_len": 0,
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


def _make_client() -> tuple[LlmClient, Mock]:
    client = LlmClient.__new__(LlmClient)
    transport = Mock()
    client._client = transport
    return client, transport


@contextmanager
def _patch_route_io(model, provider):
    provider_mock = (
        provider if isinstance(provider, AsyncMock) else AsyncMock(return_value=provider)
    )
    with (
        patch(
            "app.service.ai.llm_client.ai_model_repository.get_by_id",
            AsyncMock(return_value=model),
        ),
        patch(
            "app.service.ai.llm_client.ai_provider_repository.get_by_id",
            provider_mock,
        ),
        patch("app.service.ai.llm_client.decrypt", side_effect=lambda c: f"sk-{c}"),
    ):
        yield


def _ok_stream(text="hello"):
    lines = [
        f'data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}',
        "data: [DONE]",
    ]
    return FakeStreamResponse(200, lines)


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


def _patch_cross_services(get_call_routes, usable_keys=None):
    from app.service.ai.provider_health_service import ProviderHealthService
    from app.service.ai_model_service import AiModelService
    from app.service.ai_provider_key_service import AiProviderKeyService

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
            AiModelService, "get_call_routes", new=mocks["get_call_routes"]
        ),
        "mark_call_failed": patch.object(
            AiProviderKeyService, "mark_call_failed", new=mocks["mark_call_failed"]
        ),
        "mark_call_success": patch.object(
            AiProviderKeyService, "mark_call_success", new=mocks["mark_call_success"]
        ),
        "get_status": patch.object(ProviderHealthService, "get_status", new=mocks["get_status"]),
        "record_call": patch.object(ProviderHealthService, "record_call", new=mocks["record_call"]),
        "list_usable_keys": patch.object(
            AiProviderKeyService, "list_usable_keys", new=mocks["list_usable_keys"]
        ),
    }
    for p in patches.values():
        p.start()
    return _Services(mocks, patches)


async def test_429_switch_key(mock_redis):
    route = {"model_pk": 1, "model_id": "gpt-4o", "provider_id": 1, "model_config": {}}
    svc = _patch_cross_services([route], usable_keys=[_make_key(1), _make_key(2)])
    model = _make_model()
    provider = _make_provider()
    try:
        with _patch_route_io(model, provider):
            client, transport = _make_client()
            transport.stream = Mock(side_effect=[FakeStreamResponse(429), _ok_stream()])
            meta = {}
            chunks = [
                c
                async for c in client.stream_chat(
                    db=None,
                    redis=mock_redis,
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
    svc = _patch_cross_services(routes)
    svc.list_usable_keys.side_effect = [
        [_make_key(1, provider_id=1)],
        [_make_key(3, provider_id=2)],
    ]
    model = _make_model()
    provider_a = _make_provider(provider_id=1)
    provider_b = _make_provider(provider_id=2)
    try:
        with _patch_route_io(model, AsyncMock(side_effect=[provider_a, provider_b])):
            client, transport = _make_client()
            transport.stream = Mock(side_effect=[FakeStreamResponse(429), _ok_stream("ok")])
            meta = {}
            chunks = [
                c
                async for c in client.stream_chat(
                    db=None,
                    redis=mock_redis,
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
    svc = _patch_cross_services([route], usable_keys=[_make_key(1)])
    model = _make_model()
    provider = _make_provider()
    try:
        with _patch_route_io(model, provider):
            client, transport = _make_client()
            transport.stream = Mock(side_effect=[FakeStreamResponse(500)])
            with pytest.raises(BusinessException) as exc:
                async for _ in client.stream_chat(
                    db=None, redis=mock_redis, model_id="gpt-4o", messages=[]
                ):
                    pass
    finally:
        svc.stop()

    assert "主模型和降级模型均不可用" in str(exc.value.message)


async def test_required_caps_passed(mock_redis):
    route = {"model_pk": 1, "model_id": "gpt-4o", "provider_id": 1, "model_config": {}}
    svc = _patch_cross_services([route], usable_keys=[_make_key(1)])
    model = _make_model()
    provider = _make_provider()
    captured_caps = None
    try:
        with _patch_route_io(model, provider):
            client, transport = _make_client()
            transport.stream = Mock(side_effect=[_ok_stream()])
            async for _ in client.stream_chat(
                db=None,
                redis=mock_redis,
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
    svc = _patch_cross_services([route], usable_keys=[_make_key(1)])
    model = _make_model(
        model_id="claude-3-5-sonnet",
        provider_id=1,
        supports_prompt_cache=1,
        prompt_cache_prefix_len=100,
    )
    provider = _make_provider(provider_id=1, protocol="anthropic", auth="x-api-key")
    captured = {}

    class CaptureStream:
        async def __aenter__(self):
            return FakeStreamResponse(
                200,
                [
                    'data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}',
                    'data: {"type":"message_delta","delta":{},"usage":{"output_tokens":5}}',
                ],
            )

        async def __aexit__(self, *a):
            return False

    def _fake_stream(method, url, json=None, headers=None):
        captured["payload"] = json
        return CaptureStream()

    try:
        with _patch_route_io(model, provider):
            client, transport = _make_client()
            transport.stream = Mock(side_effect=_fake_stream)
            async for _ in client.stream_chat(
                db=None,
                redis=mock_redis,
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
    svc = _patch_cross_services([route], usable_keys=[_make_key(1), _make_key(2)])
    model = _make_model()
    provider = _make_provider()
    try:
        with _patch_route_io(model, provider):
            client, transport = _make_client()

            class _BrokenStream:
                status_code = 200

                def raise_for_status(self):
                    pass

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    return False

                async def aiter_lines(self):
                    yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
                    raise httpx.ReadError("stream broken")

            transport.stream = Mock(side_effect=[_BrokenStream(), _ok_stream("never")])
            seen = []
            with pytest.raises(BusinessException) as exc:
                async for c in client.stream_chat(
                    db=None, redis=mock_redis, model_id="gpt-4o", messages=[]
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
    svc = _patch_cross_services(routes)
    svc.list_usable_keys.return_value = [_make_key(3)]
    model = _make_model()
    provider_b = _make_provider(provider_id=2)
    try:
        async def _status(redis, provider_id):
            return "open" if provider_id == 1 else "healthy"

        svc.get_status.side_effect = _status
        with _patch_route_io(model, provider_b):
            client, transport = _make_client()
            transport.stream = Mock(side_effect=[_ok_stream("ok")])
            meta = {}
            chunks = [
                c
                async for c in client.stream_chat(
                    db=None,
                    redis=mock_redis,
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
