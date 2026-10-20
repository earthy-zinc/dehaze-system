"""Embedding 协议分派单元测试：协议族映射、端点/报文/响应解析、鉴权头（字段驱动）。

值域声明处为 `sys_ai_provider.protocol_type` 列注释 / 管理端协议下拉；
新增协议族须同步 `EMBEDDING_PROTOCOL_TYPES` 与 `get_embedding_adapter`。
"""

import json
from types import SimpleNamespace

import pytest
import respx
from httpx import Response

from app.core.exceptions import BusinessException
from app.infrastructure.embedding.embedding_dispatch import (
    EMBEDDING_PROTOCOL_TYPES,
    _cohere_embed,
    _openai_compat_embed,
    get_embedding_adapter,
)


def _provider(**kwargs):
    return SimpleNamespace(
        provider_code=kwargs.get("provider_code", "siliconflow"),
        api_base_url=kwargs.get("api_base_url", "https://api.siliconflow.cn/v1"),
        protocol_type=kwargs.get("protocol_type", "openai_compat"),
        auth_type=kwargs.get("auth_type", "bearer"),
        default_headers=kwargs.get("default_headers"),
        request_timeout_seconds=kwargs.get("request_timeout_seconds"),
    )


def test_protocol_value_domain_and_dispatch():
    assert {"openai_compat", "cohere"} == EMBEDDING_PROTOCOL_TYPES
    assert get_embedding_adapter("cohere") is _cohere_embed
    assert get_embedding_adapter("openai_compat") is _openai_compat_embed
    assert get_embedding_adapter("not_a_protocol") is _openai_compat_embed  # 未知回落


@pytest.mark.asyncio
async def test_openai_compat_endpoint_payload_and_parse():
    with respx.mock:
        route = respx.post("https://api.siliconflow.cn/v1/embeddings").mock(
            return_value=Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})
        )
        out = await _openai_compat_embed(_provider(), "sk-x", "BAAI/bge-m3", ["a", "b"])

    assert route.called
    req = route.calls.last.request
    assert req.headers["authorization"] == "Bearer sk-x"
    assert json.loads(req.content) == {"model": "BAAI/bge-m3", "input": ["a", "b"]}
    assert out == [[0.1, 0.2]]


@pytest.mark.asyncio
async def test_cohere_endpoint_and_parse():
    provider = _provider(
        provider_code="cohere", api_base_url="https://api.cohere.com", protocol_type="cohere"
    )
    with respx.mock:
        route = respx.post("https://api.cohere.com/v1/embed").mock(
            return_value=Response(200, json={"embeddings": [[0.3], [0.4]]})
        )
        out = await _cohere_embed(provider, "k", "embed-multilingual-v3.0", ["a", "b"])

    assert route.called
    assert out == [[0.3], [0.4]]


@pytest.mark.asyncio
async def test_base_url_already_contains_endpoint_not_duplicated():
    with respx.mock:
        route = respx.post("https://api.siliconflow.cn/v1/embeddings").mock(
            return_value=Response(200, json={"data": [{"embedding": [0.0]}]})
        )
        await _openai_compat_embed(
            _provider(api_base_url="https://api.siliconflow.cn/v1/embeddings"), "sk", "m", ["a"]
        )
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_auth_headers_follow_auth_type():
    with respx.mock:
        route = respx.post("https://api.siliconflow.cn/v1/embeddings").mock(
            return_value=Response(200, json={"data": [{"embedding": [0.0]}]})
        )
        await _openai_compat_embed(_provider(auth_type="x-api-key"), "sk-x", "m", ["a"])
    assert route.calls.last.request.headers["x-api-key"] == "sk-x"


@pytest.mark.asyncio
async def test_missing_base_url_raises_business_error():
    with pytest.raises(BusinessException):
        await _openai_compat_embed(_provider(api_base_url=""), "sk", "m", ["a"])
