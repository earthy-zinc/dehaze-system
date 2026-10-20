"""AI 知识库 Rerank 重排序服务

对检索候选文本按 query 相关度二次排序：端点与报文由协议分派层（rerank_dispatch）
按供应商 protocol_type 选择适配器（OpenAI 兼容 /rerank 等差异收敛在适配器内）。
模型取自知识库记录的 rerank_model，API Key 复用对应供应商体系。

失败语义：调用失败抛 BusinessException，由上层决定降级（跳过 Rerank 直接用检索原序）。
"""

import logging
from typing import Any

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.redis import get_redis_client
from app.infrastructure.provider.provider_key_selector import provider_key_selector
from app.repository.ai_provider_repository import ai_provider_repository
from app.service.kb.rerank_dispatch import get_rerank_adapter

logger = logging.getLogger(__name__)


async def _get_provider_and_key(provider_code: str):
    """从数据库选取供应商及其可用 API Key，返回 (provider, api_key)"""
    from app.database import get_db_session

    async with get_db_session() as db:
        provider = await ai_provider_repository.get_by_provider_code(db, provider_code)
        if not provider or provider.status != 1:
            raise BusinessException(
                ResultCode.AI_MODEL_NOT_AVAILABLE,
                f"Rerank 供应商 {provider_code} 不存在或已禁用",
            )
        redis = await get_redis_client()
        api_key = await provider_key_selector.select_key(db, redis, provider.id)
        if not api_key:
            raise BusinessException(
                ResultCode.AI_MODEL_NOT_AVAILABLE,
                f"Rerank 供应商 {provider_code} 无可用 API Key",
            )
        return provider, api_key


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
    provider, api_key = await _get_provider_and_key(provider_code)
    adapter = get_rerank_adapter(provider.protocol_type)
    try:
        ranked = await adapter(provider, api_key, model, query, documents, top_n)
    except BusinessException:
        raise
    except Exception as e:
        logger.warning("Rerank 调用失败(provider=%s model=%s): %s", provider_code, model, e)
        raise BusinessException(ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, "Rerank 调用失败") from e
    return [
        {
            "index": item["index"],
            "relevance": item["relevance"],
            "document": (documents[item["index"]] if 0 <= item["index"] < len(documents) else None),
        }
        for item in ranked
    ]
