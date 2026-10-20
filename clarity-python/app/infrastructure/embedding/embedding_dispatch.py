"""Embedding 协议分派：按 provider.protocol_type 选择向量化适配器（调用方不感知协议差异）

与语音分派层（infrastructure/llm/client/audio_dispatch.py）同构；统一契约（各适配器实现）：
- embed(provider, api_key, model, texts) -> list[list[float]]   批量文本 → 向量

协议实现：
- openai_compat  `POST {base}/embeddings`，报文 `{"model","input"}` → `data[].embedding`
                 （默认协议，未知协议回落此实现；OpenAI/本地/方舟/硅基流动等同构）
- cohere         `POST {base}/v1/embed`，报文 `{"model","input"}` → `embeddings[]`
                 （由 provider_code == 'cohere' 的历史特判并入协议族）

鉴权头由 `build_auth_headers` 按 provider.auth_type + default_headers 组装
（字段驱动，无 provider_code 分支）。

新增协议族须同步四处：`config/sql/schema/sys_ai_provider.sql` 的 protocol_type 列注释、
本文件的分派与 EMBEDDING_PROTOCOL_TYPES、适配器实现、设计文档协议族表，以及管理端协议下拉。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.llm.common import build_auth_headers

logger = logging.getLogger(__name__)

# 默认协议（未知协议回落此实现）
DEFAULT_EMBEDDING_PROTOCOL = "openai_compat"
PROTOCOL_COHERE = "cohere"

# embedding 协议族值域（新增协议须在此登记并补 get_embedding_adapter 分派）
EMBEDDING_PROTOCOL_TYPES = frozenset({DEFAULT_EMBEDDING_PROTOCOL, PROTOCOL_COHERE})

# 供应商未配置 request_timeout_seconds 时的默认超时（与 chat/语音口径一致）
_DEFAULT_TIMEOUT = 30.0

# 适配器统一签名：批量文本 → 与输入等长的向量列表
EmbeddingAdapter = Callable[[Any, str, str, list[str]], Awaitable[list[list[float]]]]


def _base_url(provider) -> str:
    base = (provider.api_base_url or "").rstrip("/")
    if not base:
        raise BusinessException(
            ResultCode.AI_MODEL_NOT_AVAILABLE,
            f"Embedding 供应商 {provider.provider_code} 未配置 api_base_url",
        )
    return base


def _endpoint(base: str, path: str) -> str:
    """拼接端点：base 已含该路径时原样返回（避免重复追加）"""
    return base if base.endswith(path) else base + path


async def _post(provider, api_key: str, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    timeout = provider.request_timeout_seconds or _DEFAULT_TIMEOUT
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=build_auth_headers(provider, api_key), json=payload)
        resp.raise_for_status()
        return resp.json()


async def _openai_compat_embed(
    provider, api_key: str, model: str, texts: list[str]
) -> list[list[float]]:
    """OpenAI 兼容：POST {base}/embeddings → data[].embedding"""
    url = _endpoint(_base_url(provider), "/embeddings")
    payload = {"model": model, "input": texts}
    return [item["embedding"] for item in (await _post(provider, api_key, url, payload))["data"]]


async def _cohere_embed(provider, api_key: str, model: str, texts: list[str]) -> list[list[float]]:
    """Cohere：POST {base}/v1/embed → embeddings[]"""
    url = _endpoint(_base_url(provider), "/v1/embed")
    payload = {"model": model, "input": texts}
    return (await _post(provider, api_key, url, payload))["embeddings"]


def get_embedding_adapter(protocol_type: str) -> EmbeddingAdapter:
    """按协议类型返回 embedding 适配器（未知协议回落 OpenAI 兼容实现）"""
    if protocol_type == PROTOCOL_COHERE:
        return _cohere_embed
    return _openai_compat_embed
