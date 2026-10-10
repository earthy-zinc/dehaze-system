"""AI 知识库 Embedding 向量化服务

按知识库记录的 embedding_provider / embedding_model 调用向量化端点
（模型参数来自知识库记录，而非记忆模块的全局 sys_dict 配置）。
端点与报文由协议分派层（embedding_dispatch）按供应商 protocol_type 选择适配器：
OpenAI 兼容 /embeddings 与 cohere /v1/embed 等同构差异收敛在适配器内。
提供单条与批量向量化（batch_size 分批调用）。

失败语义：向量化失败抛 BusinessException（由上层决定降级/重试），不静默返回空。
"""

import logging

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.redis import get_redis_client
from app.infrastructure.embedding.embedding_dispatch import get_embedding_adapter
from app.infrastructure.provider.provider_key_selector import provider_key_selector
from app.repository.ai_provider_repository import ai_provider_repository

logger = logging.getLogger(__name__)


async def _get_embedding_provider_and_key(provider_code: str):
    """从数据库选取 embedding 供应商及其可用 API Key，返回 (provider, api_key)"""
    from app.database import get_db_session

    async with get_db_session() as db:
        provider = await ai_provider_repository.get_by_provider_code(db, provider_code)
        if not provider or provider.status != 1:
            raise BusinessException(
                ResultCode.AI_MODEL_NOT_AVAILABLE,
                f"Embedding 供应商 {provider_code} 不存在或已禁用",
            )
        redis = await get_redis_client()
        api_key = await provider_key_selector.select_key(db, redis, provider.id)
        if not api_key:
            raise BusinessException(
                ResultCode.AI_MODEL_NOT_AVAILABLE,
                f"Embedding 供应商 {provider_code} 无可用 API Key",
            )
        return provider, api_key


async def _embed_batch(
    provider_code: str,
    model: str,
    texts: list[str],
) -> list[list[float]]:
    """调用单次 embedding 接口向量化一批文本，返回与 texts 等长的向量列表"""
    if not texts:
        return []
    provider, api_key = await _get_embedding_provider_and_key(provider_code)
    # 请求超时由供应商行显式配置（sys_ai_provider.request_timeout_seconds，NULL=30s 默认）：
    # 内置本地推理（llama.cpp 逐条串行，512-token 块 ~1.3s/条）几十块的文档单批可达 30-60s+，
    # 种子与迁移已为其置 300s；适配器内按该列取值，不再按供应商身份特判
    adapter = get_embedding_adapter(provider.protocol_type)
    try:
        return await adapter(provider, api_key, model, texts)
    except BusinessException:
        raise
    except Exception as e:
        # %r 而非 %s：httpx 超时类异常 str 为空，%s 会打出无信息量的空日志
        logger.warning("Embedding 调用失败(provider=%s model=%s): %r", provider_code, model, e)
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
