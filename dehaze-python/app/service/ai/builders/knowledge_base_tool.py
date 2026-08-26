"""知识库检索客户端（KnowledgeBaseClient）

调研结论：项目已接入平台级 AI 知识库 RAG（检索引擎 search_service，见
app/service/kb/search_service.py）。本客户端为 Agent/工具层适配：
retrieve 调用 search_service 真实检索，返回带来源引用的结果列表。

用户上下文：Agent 工具调用方（dehaze_tools_builder._knowledge_base_search）可拿到
ctx["user_id"] 时传入，检索该用户可见库（私有+公共）；无用户上下文（如网络搜索降级
路径 _degrade_to_kb）缺省检索公共库。检索失败降级返回空列表（工具层语义，不影响对话主流程）。

本地文件检索（grep/ls）由 deepagents 内置 FilesystemMiddleware 覆盖，不在此实现。
"""

import logging

logger = logging.getLogger(__name__)


class KnowledgeBaseClient:
    """AI 知识库 RAG 检索适配（Agent/工具层）"""

    async def retrieve(self, query: str, top_k: int = 5, user_id: int | None = None) -> list[dict]:
        """检索知识库，返回 [{title, snippet, source, ...}] 列表（含来源引用标识）。

        user_id 传入时检索该用户可见库（私有+公共）；缺省仅检索公共库。
        检索失败/超时降级返回空列表。
        """
        from app.database import get_db_session
        from app.dependencies.redis import get_redis_client
        from app.service.kb.search_service import search_service

        redis = await get_redis_client()
        async with get_db_session() as db:
            try:
                if user_id is not None:
                    result = await search_service.search(
                        db, redis, user_id, query, top_k=top_k
                    )
                else:
                    result = await search_service.search_internal(
                        None, query, {"topK": top_k}
                    )
            except Exception as e:  # noqa: BLE001 - 工具层降级空列表，不影响对话主流程
                logger.warning("知识库检索工具降级返回空结果 query=%s: %s", query, e)
                return []
            return [
                {
                    "title": r["documentTitle"],
                    "snippet": r["content"],
                    "source": r["documentId"],
                    "chunk_id": r["chunkId"],
                    "score": r["score"],
                }
                for r in (result.get("results") or [])
            ]

    def format_results(self, results: list[dict]) -> str:
        """将检索结果格式化为"标题+摘要+来源引用"列表文本（支持引用溯源）。"""
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            source = r.get("source") or r.get("url") or r.get("doc_id") or f"KB-{i}"
            lines.append(f"{i}. {r.get('title', '')}\n   {r.get('snippet', '')}\n   来源: {source}")
        return "\n\n".join(lines)


knowledge_base_client = KnowledgeBaseClient()
