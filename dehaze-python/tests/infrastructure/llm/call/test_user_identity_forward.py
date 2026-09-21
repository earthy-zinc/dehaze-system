"""用户身份透传运行时注入测试（AI模型管理 §2.7.8）

覆盖：
- build_user_identity 取值规则：prefix + sha256(userId)、max_len 截断、
  未启用/无配置/无用户上下文/本地 provider 不注入
- 协议落位：openai_compat 顶层字段注入（核心键不覆盖）、anthropic 嵌套路径注入
"""

import hashlib
import json
import re
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from app.infrastructure.llm.call.llm_client import LlmClient
from app.infrastructure.llm.common import build_user_identity

CHAT_URL_RE = re.compile(r"https://api\.example\.com/v1/chat/completions")
MESSAGES_URL_RE = re.compile(r"https://api\.example\.com/v1/messages")


def _expected_value(prefix="u_", user_id=42, max_len=None):
    value = prefix + hashlib.sha256(str(user_id).encode()).hexdigest()
    return value[:max_len] if max_len else value


def _make_provider(
    provider_id=1,
    protocol="openai_compat",
    provider_code="openai",
    forward=None,
):
    return SimpleNamespace(
        id=provider_id,
        provider_code=provider_code,
        protocol_type=protocol,
        auth_type="bearer",
        api_base_url="https://api.example.com/v1",
        default_headers={},
        status=1,
        user_identity_forward=forward,
    )


def _make_model(model_id="gpt-4o", pk=1, provider_id=1):
    return SimpleNamespace(
        id=pk,
        model_id=model_id,
        provider_id=provider_id,
        max_output_tokens=2048,
        supports_prompt_cache=0,
        prompt_cache_prefix_len=0,
        extra_request_params=None,
        status=1,
    )


def _make_client() -> LlmClient:
    client = LlmClient.__new__(LlmClient)
    client._client = httpx.AsyncClient()
    client._redis = None
    return client


@contextmanager
def _patch_route_io(model, provider):
    with (
        patch(
            "app.infrastructure.llm.call.llm_client.ai_model_repository.get_by_id",
            AsyncMock(return_value=model),
        ),
        patch(
            "app.infrastructure.llm.call.llm_client.ai_provider_repository.get_by_id",
            AsyncMock(return_value=provider),
        ),
        patch("app.infrastructure.llm.call.llm_client.decrypt", side_effect=lambda c: f"sk-{c}"),
    ):
        yield


def _patch_cross_services(route, usable_keys=None, redis_client=None):
    from app.infrastructure.provider.model_registry import model_registry
    from app.infrastructure.provider.provider_health_service import provider_health_service
    from app.infrastructure.provider.provider_key_selector import provider_key_selector

    patches = {
        "get_call_routes": patch.object(
            model_registry, "get_call_routes", new=AsyncMock(return_value=[route])
        ),
        "mark_call_failed": patch.object(
            provider_key_selector, "mark_call_failed", new=AsyncMock(return_value=None)
        ),
        "mark_call_success": patch.object(
            provider_key_selector, "mark_call_success", new=AsyncMock(return_value=None)
        ),
        "get_status": patch.object(
            provider_health_service, "get_status", new=AsyncMock(return_value="healthy")
        ),
        "record_call": patch.object(
            provider_health_service, "record_call", new=AsyncMock(return_value=None)
        ),
        "list_usable_keys": patch.object(
            provider_key_selector, "list_usable_keys", new=AsyncMock(return_value=usable_keys or [])
        ),
        "get_redis_client": patch(
            "app.infrastructure.llm.call.llm_client.get_redis_client",
            new=AsyncMock(return_value=redis_client),
        ),
    }
    for p in patches.values():
        p.start()
    return patches


def _stop_patches(patches):
    for p in patches.values():
        p.stop()


def _sse(lines):
    return "\n".join(lines).encode("utf-8")


def _sse_headers():
    return {"content-type": "text/event-stream"}


def _ok_lines(text="hello"):
    return [
        f'data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}',
        "data: [DONE]",
    ]


# ==================== build_user_identity 取值规则 ====================


def test_identity_prefix_plus_sha256():
    provider = _make_provider(
        forward={"enabled": True, "field": "user", "prefix": "dh_", "max_len": 512}
    )
    assert build_user_identity(provider, 42) == ("user", _expected_value(prefix="dh_"))


def test_identity_max_len_truncates():
    provider = _make_provider(
        forward={"enabled": True, "field": "user", "prefix": "u_", "max_len": 10}
    )
    assert build_user_identity(provider, 42) == ("user", _expected_value(max_len=10))


def test_identity_disabled_not_injected():
    provider = _make_provider(
        forward={"enabled": False, "field": "user", "prefix": "u_", "max_len": 64}
    )
    assert build_user_identity(provider, 42) is None


def test_identity_no_config_not_injected():
    assert build_user_identity(_make_provider(), 42) is None


def test_identity_no_user_context_not_injected():
    provider = _make_provider(
        forward={"enabled": True, "field": "user", "prefix": "u_", "max_len": 64}
    )
    assert build_user_identity(provider, None) is None


def test_identity_local_provider_not_injected():
    provider = _make_provider(
        provider_code="local",
        forward={"enabled": True, "field": "user", "prefix": "u_", "max_len": 64},
    )
    assert build_user_identity(provider, 42) is None


# ==================== openai_compat 协议落位 ====================


def _openai_forward():
    return {"enabled": True, "field": "user", "prefix": "u_", "max_len": 512}


async def test_openai_injects_top_level_user_field(mock_redis):
    route = {"model_pk": 1, "model_id": "gpt-4o", "provider_id": 1, "model_config": {}}
    patches = _patch_cross_services(
        route,
        usable_keys=[
            SimpleNamespace(id=1, key_cipher="cipher-1", daily_quota=None, rpm_limit=None)
        ],
        redis_client=mock_redis,
    )
    provider = _make_provider(forward=_openai_forward())
    captured = {}

    def _handler(request):
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content=_sse(_ok_lines()), headers=_sse_headers())

    try:
        with (
            _patch_route_io(_make_model(), provider),
            respx.mock(assert_all_mocked=True) as router,
        ):
            router.post(CHAT_URL_RE).mock(side_effect=_handler)
            client = _make_client()
            async for _ in client.stream_chat(db=None, model_id="gpt-4o", messages=[], user_id=42):
                pass
    finally:
        _stop_patches(patches)

    assert captured["payload"]["user"] == _expected_value(prefix="u_")


async def test_openai_core_key_not_overridden(mock_redis):
    route = {"model_pk": 1, "model_id": "gpt-4o", "provider_id": 1, "model_config": {}}
    patches = _patch_cross_services(
        route,
        usable_keys=[
            SimpleNamespace(id=1, key_cipher="cipher-1", daily_quota=None, rpm_limit=None)
        ],
        redis_client=mock_redis,
    )
    provider = _make_provider(
        forward={"enabled": True, "field": "model", "prefix": "u_", "max_len": 512}
    )
    captured = {}

    def _handler(request):
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content=_sse(_ok_lines()), headers=_sse_headers())

    try:
        with (
            _patch_route_io(_make_model(), provider),
            respx.mock(assert_all_mocked=True) as router,
        ):
            router.post(CHAT_URL_RE).mock(side_effect=_handler)
            client = _make_client()
            async for _ in client.stream_chat(db=None, model_id="gpt-4o", messages=[], user_id=42):
                pass
    finally:
        _stop_patches(patches)

    assert captured["payload"]["model"] == "gpt-4o"


async def test_openai_no_user_id_not_injected(mock_redis):
    route = {"model_pk": 1, "model_id": "gpt-4o", "provider_id": 1, "model_config": {}}
    patches = _patch_cross_services(
        route,
        usable_keys=[
            SimpleNamespace(id=1, key_cipher="cipher-1", daily_quota=None, rpm_limit=None)
        ],
        redis_client=mock_redis,
    )
    provider = _make_provider(forward=_openai_forward())
    captured = {}

    def _handler(request):
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content=_sse(_ok_lines()), headers=_sse_headers())

    try:
        with (
            _patch_route_io(_make_model(), provider),
            respx.mock(assert_all_mocked=True) as router,
        ):
            router.post(CHAT_URL_RE).mock(side_effect=_handler)
            client = _make_client()
            # 不传 user_id（后台调用）：透传配置存在也不注入
            async for _ in client.stream_chat(db=None, model_id="gpt-4o", messages=[]):
                pass
    finally:
        _stop_patches(patches)

    assert "user" not in captured["payload"]


# ==================== anthropic 嵌套路径落位 ====================


def _anthropic_forward():
    return {"enabled": True, "field": "metadata.user_id", "prefix": "u_", "max_len": 512}


async def test_anthropic_injects_nested_metadata_user_id(mock_redis):
    route = {"model_pk": 1, "model_id": "claude-3-5-sonnet", "provider_id": 1, "model_config": {}}
    patches = _patch_cross_services(
        route,
        usable_keys=[
            SimpleNamespace(id=1, key_cipher="cipher-1", daily_quota=None, rpm_limit=None)
        ],
        redis_client=mock_redis,
    )
    provider = _make_provider(protocol="anthropic", forward=_anthropic_forward())
    captured = {}

    def _handler(request):
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content=_sse(['data: {"type":"message_delta","delta":{},"usage":{"output_tokens":1}}']),
            headers=_sse_headers(),
        )

    try:
        with (
            _patch_route_io(_make_model("claude-3-5-sonnet"), provider),
            respx.mock(assert_all_mocked=True) as router,
        ):
            router.post(MESSAGES_URL_RE).mock(side_effect=_handler)
            client = _make_client()
            async for _ in client.stream_chat(
                db=None, model_id="claude-3-5-sonnet", messages=[], user_id=42
            ):
                pass
    finally:
        _stop_patches(patches)

    assert captured["payload"]["metadata"]["user_id"] == _expected_value(prefix="u_")


@pytest.mark.parametrize("protocol", ["openai_compat", "anthropic"])
async def test_identity_value_never_contains_raw_user_id(protocol, mock_redis):
    """合规不变量：请求体任何位置不出现明文用户 ID（哈希脱敏）"""
    route = {"model_pk": 1, "model_id": "m", "provider_id": 1, "model_config": {}}
    patches = _patch_cross_services(
        route,
        usable_keys=[
            SimpleNamespace(id=1, key_cipher="cipher-1", daily_quota=None, rpm_limit=None)
        ],
        redis_client=mock_redis,
    )
    provider = _make_provider(protocol=protocol, forward=_anthropic_forward())
    bodies = []

    def _handler(request):
        bodies.append(request.content.decode("utf-8"))
        if protocol == "anthropic":
            content = _sse(['data: {"type":"message_delta","delta":{},"usage":{}}'])
        else:
            content = _sse(_ok_lines())
        return httpx.Response(200, content=content, headers=_sse_headers())

    try:
        with (
            _patch_route_io(_make_model("m"), provider),
            respx.mock(assert_all_mocked=True) as router,
        ):
            router.post(CHAT_URL_RE if protocol == "openai_compat" else MESSAGES_URL_RE).mock(
                side_effect=_handler
            )
            client = _make_client()
            async for _ in client.stream_chat(db=None, model_id="m", messages=[], user_id=42):
                pass
    finally:
        _stop_patches(patches)

    # 请求体中不出现独立字符串 "42"（明文用户 ID），只允许哈希值
    assert bodies
    assert '"42"' not in bodies[0]
