"""AI 知识库 Rerank 重排序服务

调用 OpenAI 兼容 /rerank 端点，对检索候选文本按 query 相关度二次排序。
模型取自知识库记录的 rerank_model，API Key 复用对应供应商体系。

失败语义：调用失败抛 BusinessException，由上层决定降级（跳过 Rerank 直接用检索原序）。
"""

import logging
from typing import Any

import httpx

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.redis import get_redis_client
from app.repository.ai_provider_repository import ai_provider_repository
from app.service.ai_provider_key_service import ai_provider_key_service

logger = logging.getLogger(__name__)

# OpenAI 兼容 rerank 端点基址（按 provider_code 选择）
_RERANK_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/rerank",
    "qwen": "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
    "local": settings.MODEL_BASE_URL + "/rerank",
}


async def _get_provider_api_key(provider_code: str) -> str:
    """从数据库选取供应商的 API Key，无可用 Key 抛业务异常"""
    from app.database import get_db_session

    async with get_db_session() as db:
        provider = await ai_provider_repository.get_by_provider_code(db, provider_code)
        if not provider or provider.status != 1:
            raise BusinessException(
                ResultCode.AI_MODEL_NOT_AVAILABLE,
                f"Rerank 供应商 {provider_code} 不存在或已禁用",
            )
        redis = await get_redis_client()
        api_key = await ai_provider_key_service.select_key(db, redis, provider.id)
        if not api_key:
            raise BusinessException(
                ResultCode.AI_MODEL_NOT_AVAILABLE,
                f"Rerank 供应商 {provider_code} 无可用 API Key",
            )
        return api_key


async def rerank(
    provider_code: str,
    model: str,
    query: str,
    documents: list[str],
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """对候选文本列表按 query 重排序，返回按分数降序的 [{index, relevance, document}] 列表。

    top_n 缺省时返回全部候选的排序结果。
    """
    if not documents:
        return []
    url = _RERANK_ENDPOINTS.get(provider_code)
    if not url:
        raise BusinessException(
            ResultCode.AI_MODEL_NOT_AVAILABLE,
            f"未知 rerank provider_code={provider_code}",
        )
    api_key = await _get_provider_api_key(provider_code)
    payload: dict[str, Any] = {"model": model, "query": query, "documents": documents}
    if top_n:
        payload["top_n"] = top_n
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            results = data.get("results") or []
            ordered: list[dict[str, Any]] = []
            for item in sorted(results, key=lambda r: r.get("relevance_score", 0.0), reverse=True):
                ordered.append(
                    {
                        "index": item.get("index"),
                        "relevance": item.get("relevance_score"),
                        "document": (
                            documents[item["index"]]
                            if 0 <= item["index"] < len(documents)
                            else None
                        ),
                    }
                )
            return ordered
    except Exception as e:
        logger.warning("Rerank 调用失败(provider=%s model=%s): %s", provider_code, model, e)
        raise BusinessException(
            ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, "Rerank 调用失败"
        ) from e
