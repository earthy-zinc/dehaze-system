"""Elasticsearch 客户端封装

提供连接管理、索引 CRUD、向量检索与全文检索能力。
ES 为必选基础设施（docker-compose 统一部署）；运行期故障（连接失败等）时方法
返回 None / 空列表并记 warning，不抛异常，避免打断对话主流程。

连接由实例持有（es_client 模块级单例），测试 monkeypatch 实例的 _client
属性即可完成隔离。
"""

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


class EsClient:
    """ES 客户端：连接管理 + 索引 CRUD + 向量检索（es_client 单例）"""

    def __init__(self) -> None:
        self._client: Any = None

    async def get_client(self) -> Any:
        """获取 ES 客户端，初始化失败返回 None"""
        if self._client is None:
            try:
                from elasticsearch import AsyncElasticsearch

                kwargs: dict = {"hosts": [settings.ES_URL]}
                if settings.ES_API_KEY:
                    kwargs["api_key"] = settings.ES_API_KEY
                elif settings.ES_USERNAME:
                    kwargs["basic_auth"] = (settings.ES_USERNAME, settings.ES_PASSWORD)
                self._client = AsyncElasticsearch(**kwargs)
            except Exception as e:
                logger.warning("ES 客户端初始化失败: %s", e)
                return None
        return self._client

    async def ensure_index(self, index_name: str, mappings: dict) -> bool:
        """确保索引存在，不存在则创建"""
        client = await self.get_client()
        if client is None:
            return False
        try:
            if not await client.indices.exists(index=index_name):
                await client.indices.create(index=index_name, mappings=mappings)
            return True
        except Exception as e:
            logger.warning("ES 创建索引 %s 失败: %s", index_name, e)
            return False

    async def index_doc(self, index_name: str, doc_id: str, doc: dict) -> bool:
        """索引文档"""
        client = await self.get_client()
        if client is None:
            return False
        try:
            await client.index(index=index_name, id=doc_id, document=doc, refresh=True)
            return True
        except Exception as e:
            logger.warning("ES 索引文档 %s/%s 失败: %s", index_name, doc_id, e)
            return False

    async def delete_doc(self, index_name: str, doc_id: str) -> bool:
        """删除索引文档（记忆删除时同步清除向量，避免残留）"""
        client = await self.get_client()
        if client is None:
            return False
        try:
            await client.delete(index=index_name, id=doc_id, refresh=True)
            return True
        except Exception as e:
            logger.warning("ES 删除文档 %s/%s 失败: %s", index_name, doc_id, e)
            return False

    async def vector_search(
        self,
        index_name: str,
        query_vector: list[float],
        filters: list[dict],
        top_n: int = 5,
    ) -> list[dict]:
        """向量相似度检索。

        返回命中文档 _source 列表，并在每条中注入 ES 相似度分数 ``relevance``
        （kNN _score，供下游三维权重排序做归一化），字段冲突时以命中分数为准。
        """
        client = await self.get_client()
        if client is None:
            return []
        try:
            body = {
                "size": top_n,
                "query": {
                    "bool": {
                        "filter": filters,
                        "must": [
                            {
                                "knn": {
                                    "field": "content_vector",
                                    "query_vector": query_vector,
                                    "k": top_n,
                                    "num_candidates": top_n * 10,
                                }
                            }
                        ],
                    }
                },
            }
            resp = await client.search(index=index_name, body=body)
            hits = []
            for hit in resp["hits"]["hits"]:
                doc = dict(hit["_source"])
                doc["relevance"] = hit.get("_score", 0.0)
                hits.append(doc)
            return hits
        except Exception as e:
            logger.warning("ES 向量检索失败: %s", e)
            return []

    async def text_search(
        self,
        index_name: str,
        query: str,
        filters: list[dict],
        top_n: int = 20,
    ) -> list[dict]:
        """全文检索，返回命中文档 _source 列表"""
        hits, _ = await self.paged_text_search(index_name, query, filters, page=1, size=top_n)
        return hits

    async def paged_text_search(
        self,
        index_name: str,
        query: str,
        filters: list[dict],
        page: int = 1,
        size: int = 20,
    ) -> tuple[list[dict], int]:
        """全文检索（分页），返回 (命中文档 _source 列表, 命中总数)"""
        client = await self.get_client()
        if client is None:
            return [], 0
        try:
            body = {
                "from": (page - 1) * size,
                "size": size,
                "query": {
                    "bool": {
                        "filter": filters,
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["title^3", "message_contents"],
                                }
                            }
                        ],
                    }
                },
            }
            resp = await client.search(index=index_name, body=body)
            hits = [hit["_source"] for hit in resp["hits"]["hits"]]
            total = int(resp["hits"]["total"]["value"])
            return hits, total
        except Exception as e:
            logger.warning("ES 全文检索失败: %s", e)
            return [], 0


es_client = EsClient()
