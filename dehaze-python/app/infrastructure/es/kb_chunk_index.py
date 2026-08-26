"""知识库分块 ES 向量索引定义与读写原语（技术适配层）

索引名：kb_chunks_{knowledgeBaseId}（按知识库分索引隔离）
字段：content_vector(dense_vector/cosine)、content(text 全文)、doc_title(text)、
doc_id/chunk_id/chunk_index/version、create_time(date)、metadata(object 动态+显式过滤子字段)、
tags(keyword)。metadata 显式定义过滤用子字段（type/algorithm_id/entities.name/relations.type），
其余 metadata 内容保持动态映射。

只承载索引定义、ensure/delete、批量写入/删除、向量/关键词/RRF 混合检索原语；
Embedding 计算、配置读取、API Key 选择等业务编排见 service 层 kb 服务，勿在此引入 httpx/repository。
"""

import logging
from datetime import datetime

from app.infrastructure.es.es_client import es_client

logger = logging.getLogger(__name__)

INDEX_PREFIX = "kb_chunks_{kb_id}"


def kb_index_name(kb_id: int) -> str:
    return INDEX_PREFIX.format(kb_id=kb_id)


def _kb_mappings(dims: int) -> dict:
    return {
        "properties": {
            "content_vector": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "cosine",
            },
            # ik 中文分词（analysis-ik 插件，es-plugin-init 自动安装）：
            # analyzer=ik_max_word 索引侧最大粒度提召回，search_analyzer=ik_smart 查询侧粗粒度
            "content": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
            "doc_title": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
            "doc_id": {"type": "long"},
            "chunk_id": {"type": "long"},
            "chunk_index": {"type": "integer"},
            "version": {"type": "integer"},
            "create_time": {"type": "date"},
            # 过滤用子字段显式声明为 keyword/long，保证 term/terms 查询精确匹配；
            # 其余 metadata 内容（page/paragraph/entities/relations 详情等）保持动态映射
            "metadata": {
                "type": "object",
                "dynamic": True,
                "properties": {
                    "type": {"type": "keyword"},
                    "algorithm_id": {"type": "long"},
                    "entities": {
                        "type": "object",
                        "properties": {"name": {"type": "keyword"}},
                    },
                    "relations": {
                        "type": "object",
                        "properties": {"type": {"type": "keyword"}},
                    },
                },
            },
            "tags": {"type": "keyword"},
        }
    }


async def ensure_kb_index(kb_id: int, dims: int) -> bool:
    """确保知识库索引存在（mapping dims 由调用方按该库 embedding 模型维度传入）"""
    return await es_client.ensure_index(kb_index_name(kb_id), _kb_mappings(dims))


async def delete_kb_index(kb_id: int) -> bool:
    """删除知识库索引（知识库删除时同步清理向量，避免残留）"""
    client = await es_client.get_client()
    if client is None:
        return False
    index = kb_index_name(kb_id)
    try:
        if await client.indices.exists(index=index):
            await client.indices.delete(index=index)
        return True
    except Exception as e:  # noqa: BLE001 - 索引清理失败仅告警，不影响主流程
        logger.warning("ES 删除索引 %s 失败: %s", index, e)
        return False


async def bulk_index_chunks(kb_id: int, docs: list[dict]) -> bool:
    """批量写入分块到 ES 索引（docs 为已含 content_vector 的分块字典列表）"""
    client = await es_client.get_client()
    if client is None or not docs:
        return False
    index = kb_index_name(kb_id)
    actions = []
    for doc in docs:
        actions.append({"index": {"_index": index, "_id": str(doc["chunk_id"])}})
        actions.append(doc)
    try:
        resp = await client.bulk(operations=actions)
        if resp.get("errors"):
            # 部分写入失败：记录失败项明细并返回 False，供上层流水线触发重试
            failed_ids = []
            for item in resp.get("items") or []:
                index_result = item.get("index") or {}
                if index_result.get("status", 200) >= 300 or index_result.get("error"):
                    failed_ids.append(
                        f"{index_result.get('_id')}:{index_result.get('error')}"
                    )
            logger.warning("ES 批量写入 %s 存在失败项: %s", index, failed_ids)
            return False
        return True
    except Exception as e:  # noqa: BLE001 - 批量写失败由上层补偿重试
        logger.warning("ES 批量写入 %s 失败: %s", index, e)
        return False


async def delete_doc_chunks(kb_id: int, document_id: int) -> bool:
    """删除文档下所有分块索引（文档删除/更新替换旧版本分块时清除，避免残留）"""
    client = await es_client.get_client()
    if client is None:
        return False
    index = kb_index_name(kb_id)
    try:
        # conflicts=proceed：并发删除遇版本冲突时跳过冲突项继续，避免整批中断
        await client.delete_by_query(
            index=index,
            query={"term": {"doc_id": document_id}},
            refresh=True,
            conflicts="proceed",
        )
        return True
    except Exception as e:  # noqa: BLE001 - 删除失败由对账任务补偿
        logger.warning("ES 删除文档 %s 分块失败: %s", document_id, e)
        return False


async def get_index_stats(kb_id: int) -> dict:
    """查询知识库索引大小与文档数（管理端索引状态）。

    经 indices.stats 的 store/documents 指标取真实存储占用（含副本）与分块文档数；
    索引不存在（知识库创建后未写入分块时索引可能不存在）返回零值，不报错。
    """
    client = await es_client.get_client()
    if client is None:
        return {"index_size": 0, "index_doc_count": 0}
    index = kb_index_name(kb_id)
    try:
        resp = await client.indices.stats(index=index, metric="store,docs")
        indices = (resp.get("indices") or {}).get(index) or {}
        store = indices.get("total") or {}
        docs = indices.get("total") or {}
        return {
            "index_size": int(store.get("store", {}).get("size_in_bytes", 0) or 0),
            "index_doc_count": int(docs.get("docs", {}).get("count", 0) or 0),
        }
    except Exception as e:  # noqa: BLE001 - 索引不存在或查询失败按空索引处理
        logger.warning("ES 索引统计 %s 失败: %s", index, e)
        return {"index_size": 0, "index_doc_count": 0}


def build_filters(
    *,
    doc_type: str | None = None,
    tags: list[str] | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    algorithm_id: int | None = None,
    entities: list[str] | None = None,
    relations: list[str] | None = None,
) -> list[dict]:
    """构建元数据 filter 子句（检索前缩小范围）。

    支持文档类型/标签/时间范围/算法关联/实体/关系维度过滤。
    metadata 为 object 动态字段，算法与实体/关系分别落在 metadata.algorithm_id 与
    metadata.entities/relations（知识图谱阶段 1 结构）。
    """
    filters: list[dict] = []
    if doc_type:
        filters.append({"term": {"metadata.type": doc_type}})
    if tags:
        filters.append({"terms": {"tags": tags}})
    if start_time or end_time:
        range_clause: dict = {}
        if start_time:
            range_clause["gte"] = start_time.isoformat()
        if end_time:
            range_clause["lte"] = end_time.isoformat()
        filters.append({"range": {"create_time": range_clause}})
    if algorithm_id:
        filters.append({"term": {"metadata.algorithm_id": algorithm_id}})
    if entities:
        filters.append({"terms": {"metadata.entities.name": entities}})
    if relations:
        filters.append({"terms": {"metadata.relations.type": relations}})
    return filters


async def vector_search(
    kb_id: int,
    query_vector: list[float],
    filters: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """纯向量检索（ES 8.12+ 顶层 knn 语法），返回命中文档 _source 列表（含 relevance 分数）。

    使用顶层 knn 的 filter 作为 filtered HNSW 前置过滤，保证过滤后仍返回足额 top_n。
    """
    body: dict = {
        "size": top_n,
        "knn": {
            "field": "content_vector",
            "query_vector": query_vector,
            "k": top_n,
            "num_candidates": top_n * 10,
            "filter": {"bool": {"filter": filters}},
        },
    }
    return await _search(kb_index_name(kb_id), body)


async def keyword_search(
    kb_id: int,
    query: str,
    filters: list[dict],
    top_n: int = 20,
) -> list[dict]:
    """纯关键词(BM25)检索，返回命中文档 _source 列表（含 relevance 分数）"""
    body = {
        "size": top_n,
        "query": {
            "bool": {
                "filter": filters,
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["content", "doc_title^2"],
                        }
                    }
                ],
            }
        },
    }
    return await _search(kb_index_name(kb_id), body)


async def hybrid_search(
    kb_id: int,
    query: str,
    query_vector: list[float],
    filters: list[dict],
    top_k: int = 5,
    rank_constant: int = 60,
    rank_window_size: int = 20,
    vector_weight: float = 1.0,
    keyword_weight: float = 1.0,
) -> list[dict]:
    """RRF 混合检索（ES 8.x retriever API）：kNN retriever + standard retriever 融合。

    过滤条件分别注入各子 retriever（knn retriever 的 filter + standard retriever 的
    bool 外层），避免顶层 filter 与 retriever 并用的版本兼容风险。
    返回命中文档 _source 列表（含 relevance=RRF 融合分数）。
    """
    filter_clause = {"bool": {"filter": filters}} if filters else None
    keyword_query: dict = {
        "multi_match": {"query": query, "fields": ["content", "doc_title^2"]}
    }
    if filter_clause:
        keyword_query = {"bool": {"filter": filters, "must": [keyword_query]}}
    retrievers: list[dict] = [
        {
            "weight": vector_weight,
            "knn": {
                "field": "content_vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": top_k * 10,
                **({"filter": filter_clause} if filter_clause else {}),
            },
        },
        {
            "weight": keyword_weight,
            "standard": {"query": keyword_query},
        },
    ]
    body: dict = {
        "size": top_k,
        "retriever": {
            "rrf": {
                "retrievers": retrievers,
                "rank_constant": rank_constant,
                "rank_window_size": rank_window_size,
            }
        },
    }
    return await _search(kb_index_name(kb_id), body)


async def _search(index: str, body: dict) -> list[dict]:
    """执行检索并提取 _source，注入 relevance 分数"""
    client = await es_client.get_client()
    if client is None:
        return []
    try:
        resp = await client.search(index=index, body=body)
        hits = []
        for hit in resp["hits"]["hits"]:
            doc = dict(hit["_source"])
            doc["relevance"] = hit.get("_score", 0.0)
            hits.append(doc)
        return hits
    except Exception as e:  # noqa: BLE001 - 检索失败由调用方降级（无知识回复）
        logger.warning("ES 检索 %s 失败: %s", index, e)
        return []
