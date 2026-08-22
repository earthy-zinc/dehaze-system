"""记忆 ES 读模型服务（CQRS 读模型编排层）

职责：承载"记忆向量索引"的业务编排——Embedding 计算（外部 API）、配置读取
（sys_dict ai_embedding）、API Key 选择、以及 sync/search 的聚合策略。底层
索引定义与读写/检索原语由 es/ai_memory_index 提供。

调用方错误语义约定：
- 记忆保存链路（memory_extraction.save_extracted_memories）：Embedding 失败返回空
  向量时静默跳过 ES 同步（不抛异常，ES 未启用降级 MySQL LIKE）。
- 记忆检索链路（memory_injection）：Embedding/检索失败返回空列表，由调用方降级
  MySQL LIKE。
"""

import logging
from typing import Any

import httpx

from app.config import settings
from app.dependencies.redis import get_redis_client
from app.infrastructure.es.ai_memory_index import (
    DEFAULT_DIMS,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
)
from app.infrastructure.es.ai_memory_index import (
    ensure_memory_index as _es_ensure_memory_index,
)
from app.infrastructure.es.ai_memory_index import (
    search_memories as _es_search_memories,
)
from app.infrastructure.es.ai_memory_index import (
    sync_memory_doc as _es_sync_memory_doc,
)
from app.repository.ai_provider_repository import ai_provider_repository
from app.repository.dict_repository import dict_repository
from app.service.ai_provider_key_service import AiProviderKeyService

logger = logging.getLogger(__name__)

# Embedding 配置字典类型（provider_code/model/dims 三键种子，见 config/sql/data/sys_dict.sql）
_EMBEDDING_DICT = "ai_embedding"

# OpenAI 兼容 embedding 端点基址（按 provider_code 选择，其余供应商暂走 OpenAI 兼容协议）
_EMBEDDING_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/embeddings",
    "qwen": "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
    "cohere": "https://api.cohere.com/v1/embed",
}


async def _load_embedding_config() -> dict[str, Any]:
    """从 sys_dict（ai_embedding）读取 embedding 配置，缺省回落种子默认。"""
    config: dict[str, Any] = {
        "provider_code": DEFAULT_PROVIDER,
        "model": DEFAULT_MODEL,
        "dims": DEFAULT_DIMS,
    }
    try:
        from app.database import get_db_session

        async with get_db_session() as db:
            items = await dict_repository.list_enabled_by_type_code(db, _EMBEDDING_DICT)
            for item in items:
                config[item.name] = item.value
    except Exception as e:  # noqa: BLE001 - 读取失败回落种子默认，不影响降级可用
        logger.warning("读取 embedding 配置失败，使用种子默认: %s", e)
    return config


async def _get_embedding_api_key() -> str:
    """从数据库选取 embedding 供应商的 API Key，无可用 Key 返回空串"""
    from app.database import get_db_session

    config = await _load_embedding_config()
    async with get_db_session() as db:
        provider = await ai_provider_repository.get_by_provider_code(db, config["provider_code"])
        if not provider or provider.status != 1:
            return ""
        redis = await get_redis_client()
        return await AiProviderKeyService.select_key(db, redis, provider.id) or ""


async def get_embedding(text: str) -> list[float]:
    """调用 embedding 模型获取向量，失败返回空列表"""
    if not settings.ES_ENABLED:
        return []
    config = await _load_embedding_config()
    api_key = await _get_embedding_api_key()
    if not api_key:
        return []
    url = _EMBEDDING_ENDPOINTS.get(config["provider_code"])
    if not url:
        logger.warning("未知 embedding provider_code=%s", config["provider_code"])
        return []
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": config["model"], "input": text},
            )
            resp.raise_for_status()
            data = resp.json()
            # OpenAI 兼容返回 data[0].embedding；cohere 返回 embeddings[0]
            if "embeddings" in data:
                return data["embeddings"][0]
            return data["data"][0]["embedding"]
    except Exception as e:
        logger.warning("Embedding 调用失败: %s", e)
        return []


async def ensure_memory_index() -> bool:
    """确保记忆索引存在（mapping dims 取当前 embedding 配置维度）"""
    config = await _load_embedding_config()
    dims = int(config.get("dims") or DEFAULT_DIMS)
    return await _es_ensure_memory_index(dims)


async def sync_memory(memory: dict) -> bool:
    """计算 embedding 并同步单条记忆到 ES"""
    vector = await get_embedding(memory["content"])
    if not vector:
        return False
    return await _es_sync_memory_doc(memory, vector)


async def search_memories(user_id: int, query: str, top_n: int = 5) -> list[dict]:
    """向量检索记忆（计算 query embedding 后检索），返回记忆 dict 列表"""
    vector = await get_embedding(query)
    if not vector:
        return []
    return await _es_search_memories(vector, user_id, top_n)
