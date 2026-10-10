"""Rerank 协议分派：按 provider.protocol_type 选择重排序适配器（调用方不感知协议差异）

与语音/Embedding 分派层同构；统一契约（各适配器实现）：
- rerank(provider, api_key, model, query, documents, top_n) -> list[dict]
  返回按相关度降序的 `[{"index": int, "relevance": float}]`（index 为候选在 documents 中的下标）

协议实现：
- openai_compat  `POST {base}/rerank`，报文 `{"model","query","documents"[,"top_n"]}`
                 → `results[].relevance_score`（默认协议，未知协议回落此实现）
- volc_vikingdb  火山 VikingDB 知识库 rerank（`/api/knowledge/service/rerank`、
                 `datas[].query/content`、`data.scores`、AK/SK 签名）——**待接入**：
                 签名型厂商需双因子凭据（AccessKeyId + SecretAccessKey），而
                 `sys_ai_provider_key` 为单密钥模型，须先扩展凭据模型再实现适配器。

鉴权头由 `build_auth_headers` 按 provider.auth_type + default_headers 组装
（字段驱动，无 provider_code 分支）。

新增协议族须同步四处：`config/sql/schema/sys_ai_provider.sql` 的 protocol_type 列注释、
本文件的分派与 RERANK_PROTOCOL_TYPES、适配器实现、设计文档协议族表，以及管理端协议下拉。
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
DEFAULT_RERANK_PROTOCOL = "openai_compat"
# 待接入协议族（签名型厂商，见模块 docstring）
PROTOCOL_VOLC_VIKINGDB = "volc_vikingdb"

# rerank 协议族值域（新增协议须在此登记并补 get_rerank_adapter 分派）
RERANK_PROTOCOL_TYPES = frozenset({DEFAULT_RERANK_PROTOCOL})

# 供应商未配置 request_timeout_seconds 时的默认超时
_DEFAULT_TIMEOUT = 30.0

# 适配器统一签名：候选文本 → 按相关度降序的 [{index, relevance}]
RerankAdapter = Callable[[Any, str, str, str, list[str], "int | None"], Awaitable[list[dict]]]


def _base_url(provider) -> str:
    base = (provider.api_base_url or "").rstrip("/")
    if not base:
        raise BusinessException(
            ResultCode.AI_MODEL_NOT_AVAILABLE,
            f"Rerank 供应商 {provider.provider_code} 未配置 api_base_url",
        )
    return base


def _endpoint(base: str, path: str) -> str:
    """拼接端点：base 已含该路径时原样返回（避免重复追加）"""
    return base if base.endswith(path) else base + path


async def _openai_compat_rerank(
    provider, api_key: str, model: str, query: str, documents: list[str], top_n: int | None
) -> list[dict]:
    """OpenAI 兼容：POST {base}/rerank → results[].relevance_score"""
    payload: dict[str, Any] = {"model": model, "query": query, "documents": documents}
    if top_n:
        payload["top_n"] = top_n
    timeout = provider.request_timeout_seconds or _DEFAULT_TIMEOUT
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            _endpoint(_base_url(provider), "/rerank"),
            headers=build_auth_headers(provider, api_key),
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    results = data.get("results") or []
    return [
        {"index": item.get("index"), "relevance": item.get("relevance_score")}
        for item in sorted(results, key=lambda r: r.get("relevance_score", 0.0), reverse=True)
    ]


def get_rerank_adapter(protocol_type: str) -> RerankAdapter:
    """按协议类型返回 rerank 适配器（未知协议回落 OpenAI 兼容实现）"""
    return _openai_compat_rerank
