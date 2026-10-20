"""Rerank 协议分派单元测试：协议族映射、端点/报文/解析 + 服务层文档映射与排序。

值域声明处为 `sys_ai_provider.protocol_type` 列注释 / 管理端协议下拉；
新增协议族须同步 `RERANK_PROTOCOL_TYPES` 与 `get_rerank_adapter`。
"""

import json
from types import SimpleNamespace

import pytest
import respx
from httpx import Response

from app.service.kb import rerank_service
from app.service.kb.rerank_dispatch import (
    RERANK_PROTOCOL_TYPES,
    _openai_compat_rerank,
    get_rerank_adapter,
)


def _provider(**kwargs):
    return SimpleNamespace(
        provider_code=kwargs.get("provider_code", "siliconflow"),
        api_base_url=kwargs.get("api_base_url", "https://api.siliconflow.cn/v1"),
        protocol_type=kwargs.get("protocol_type", "openai_compat"),
        auth_type=kwargs.get("auth_type", "bearer"),
        default_headers=None,
        request_timeout_seconds=kwargs.get("request_timeout_seconds"),
    )


def test_protocol_value_domain_and_dispatch():
    assert {"openai_compat"} == RERANK_PROTOCOL_TYPES
    assert get_rerank_adapter("openai_compat") is _openai_compat_rerank
    assert get_rerank_adapter("not_a_protocol") is _openai_compat_rerank  # 未知回落


@pytest.mark.asyncio
async def test_openai_compat_endpoint_payload_and_sorted_desc():
    with respx.mock:
        route = respx.post("https://api.siliconflow.cn/v1/rerank").mock(
            return_value=Response(
                200,
                json={
                    "results": [
                        {"index": 1, "relevance_score": 0.2},
                        {"index": 0, "relevance_score": 0.9},
                    ]
                },
            )
        )
        ranked = await _openai_compat_rerank(
            _provider(), "sk-x", "BAAI/bge-reranker-v2-m3", "q", ["a", "b"], 2
        )

    assert route.called
    req = route.calls.last.request
    assert req.headers["authorization"] == "Bearer sk-x"
    assert json.loads(req.content) == {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "q",
        "documents": ["a", "b"],
        "top_n": 2,
    }
    # 按相关度降序
    assert ranked == [{"index": 0, "relevance": 0.9}, {"index": 1, "relevance": 0.2}]


@pytest.mark.asyncio
async def test_rerank_service_maps_documents_and_drops_out_of_range(monkeypatch):
    async def _provider_and_key(provider_code):
        return _provider(), "sk-x"

    monkeypatch.setattr(rerank_service, "_get_provider_and_key", _provider_and_key)
    with respx.mock:
        respx.post("https://api.siliconflow.cn/v1/rerank").mock(
            return_value=Response(
                200,
                json={
                    "results": [
                        {"index": 1, "relevance_score": 0.5},
                        {"index": 99, "relevance_score": 0.9},
                    ]
                },
            )
        )
        out = await rerank_service.rerank("siliconflow", "r1", "q", ["a", "b"])

    assert out[0]["index"] == 99
    assert out[0]["document"] is None  # 越界 index 不取文档
    assert out[1]["document"] == "b"


@pytest.mark.asyncio
async def test_rerank_service_empty_documents_skips_call():
    out = await rerank_service.rerank("siliconflow", "r1", "q", [])
    assert out == []
