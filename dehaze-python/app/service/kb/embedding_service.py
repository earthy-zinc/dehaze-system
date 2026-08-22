"""AI 知识库 Embedding 向量化服务

按知识库记录的 embedding_provider / embedding_model 调用 OpenAI 兼容端点向量化
（模型参数来自知识库记录，而非记忆模块的全局 sys_dict 配置）。
提供单条与批量向量化（batch_size 分批调用）、常用模型维度映射。

失败语义：向量化失败抛 BusinessException（由上层决定降级/重试），不静默返回空。
"""

import logging
from typing import Any

import httpx

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.redis import get_redis_client
from app.repository.ai_provider_repository import ai_provider_repository
from app.service.ai_provider_key_service import AiProviderKeyService

logger = logging.getLogger(__name__)

# OpenAI 兼容 embedding 端点基址（按 provider_code 选择，其余供应商暂走 OpenAI 兼容协议）
_EMBEDDING_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/embeddings",
    "qwen": "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
    "cohere": "https://api.cohere.com/v1/embed",
    # 内置本地 LLM 服务（Qwen3-Embedding-0.6B，1024 维，与 bge-m3 同维度）
    "local": f"http://{settings.LOCAL_LLM_HOST}:{settings.LOCAL_LLM_PORT}/v1/embeddings",
}

# 常用模型 -> 向量维度映射（ES dense_vector dims 联动，未知模型时由供应商配置下发）
_KNOWN_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "bge-m3": 1024,
    "bge-large-zh": 1024,
}


def get_embedding_dim(provider: str, model: str) -> int:
    """返回常用 embedding 模型的向量维度（未知模型返回 0，由调用方读取供应商配置 dims）"""
    return _KNOWN_DIMS.get(model, 0)


async def _get_embedding_api_key(provider_code: str) -> str:
    """从数据库选取 embedding 供应商的 API Key，无可用 Key 抛业务异常"""
    from app.database import get_db_session

    async with get_db_session() as db:
        provider = await ai_provider_repository.get_by_provider_code(db, provider_code)
        if not provider or provider.status != 1:
            raise BusinessException(
                ResultCode.AI_MODEL_NOT_AVAILABLE,
                f"Embedding 供应商 {provider_code} 不存在或已禁用",
            )
        redis = await get_redis_client()
        api_key = await AiProviderKeyService.select_key(db, redis, provider.id)
        if not api_key:
            raise BusinessException(
                ResultCode.AI_MODEL_NOT_AVAILABLE,
                f"Embedding 供应商 {provider_code} 无可用 API Key",
            )
        return api_key


async def _embed_batch(
    provider_code: str,
    model: str,
    texts: list[str],
) -> list[list[float]]:
    """调用单次 embedding 接口向量化一批文本，返回与 texts 等长的向量列表"""
    if not texts:
        return []
    url = _EMBEDDING_ENDPOINTS.get(provider_code)
    if not url:
        raise BusinessException(
            ResultCode.AI_MODEL_NOT_AVAILABLE,
            f"未知 embedding provider_code={provider_code}",
        )
    api_key = await _get_embedding_api_key(provider_code)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "input": texts},
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            # OpenAI 兼容返回 data[].embedding；cohere 返回 embeddings[]
            if "embeddings" in data:
                return data["embeddings"]
            return [item["embedding"] for item in data["data"]]
    except Exception as e:
        logger.warning("Embedding 调用失败(provider=%s model=%s): %s", provider_code, model, e)
        raise BusinessException(
            ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, "Embedding 调用失败"
        ) from e


async def embed_texts(
    provider_code: str,
    model: str,
    texts: list[str],
    batch_size: int = 100,
) -> list[list[float]]:
    """批量向量化文本，按 batch_size 分批调用，返回与 texts 顺序一致的向量列表"""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        vectors.extend(await _embed_batch(provider_code, model, texts[i : i + batch_size]))
    return vectors


async def embed_text(provider_code: str, model: str, text: str) -> list[float]:
    """向量化单条文本"""
    vectors = await embed_texts(provider_code, model, [text])
    return vectors[0] if vectors else []
