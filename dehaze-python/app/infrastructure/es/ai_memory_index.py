"""记忆 ES 向量索引定义与读写原语（技术适配层）

索引名：ai_memory
字段：id, user_id, memory_type, content, content_vector, importance, status, archived, deleted
只承载索引定义、ensure（dims 由调用方传入）、纯文档写入/删除与向量检索原语；
Embedding 计算、配置读取（sys_dict）、API Key 选择等业务编排见 service 层
memory_es_service，勿在此引入 httpx / repository / db session。
"""

import logging

from app.infrastructure.es.es_client import es_client

logger = logging.getLogger(__name__)

INDEX_NAME = "ai_memory"

# Embedding 缺省回落值（与 config/sql/data/sys_dict.sql 的 ai_embedding 种子同源，
# 供 seed 契约测试校验；实际配置读取见 service 层 memory_es_service）
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIMS = 1536


async def ensure_memory_index(dims: int) -> bool:
    """确保记忆索引存在（mapping dims 由调用方按当前 embedding 配置传入）"""
    mappings = {
        "properties": {
            "id": {"type": "long"},
            "user_id": {"type": "long"},
            "memory_type": {"type": "keyword"},
            "content": {"type": "text"},
            "content_vector": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "cosine",
            },
            "importance": {"type": "integer"},
            "status": {"type": "integer"},
            "archived": {"type": "integer"},
            "deleted": {"type": "integer"},
        }
    }
    return await es_client.ensure_index(INDEX_NAME, mappings)


async def sync_memory_doc(memory: dict, vector: list[float]) -> bool:
    """写入/更新单条记忆到 ES（纯文档写入，vector 由调用方计算）"""
    doc = {
        "id": memory["id"],
        "user_id": memory["user_id"],
        "memory_type": memory["memory_type"],
        "content": memory["content"],
        "content_vector": vector,
        "importance": memory["importance"],
        "status": memory["status"],
        "archived": memory["archived"],
        "deleted": memory["deleted"],
    }
    return await es_client.index_doc(INDEX_NAME, str(memory["id"]), doc)


async def delete_memory_doc(memory_id: int) -> bool:
    """删除单条记忆的 ES 文档（记忆软删时同步清除向量索引）"""
    return await es_client.delete_doc(INDEX_NAME, str(memory_id))


async def search_memories(vector: list[float], user_id: int, top_n: int = 5) -> list[dict]:
    """向量检索记忆（纯检索原语，vector 由调用方计算），返回记忆 dict 列表"""
    filters = [
        {"term": {"user_id": user_id}},
        {"term": {"status": 1}},
        {"term": {"archived": 0}},
        {"term": {"deleted": 0}},
    ]
    return await es_client.vector_search(INDEX_NAME, vector, filters, top_n)
