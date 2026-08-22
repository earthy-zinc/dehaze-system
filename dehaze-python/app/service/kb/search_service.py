"""AI 知识库检索引擎 Service

对齐《后端实现-检索引擎.md》§2/§3/§4/§5/§6：多库可见性过滤 → 结果缓存 →
多库 embedding 分组向量化 → 按 search_strategy 分派 vector/keyword/hybrid →
阈值过滤 → Rerank → MMR 多样性 → 超时降级。

返回结构（§5.1）：{query, knowledgeBaseIds, results: [{chunkId, content, score,
documentTitle, documentId, chunkIndex, metadata}]}
"""

import asyncio
import hashlib
import json
import logging
import math
from collections import defaultdict

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.es import kb_chunk_index
from app.infrastructure.es.kb_chunk_index import build_filters
from app.models.entity.sys_knowledge_base import SysKnowledgeBase
from app.repository.knowledge_base_repository import knowledge_base_repository
from app.service.kb.embedding_service import embed_text
from app.service.kb.rerank_service import rerank

logger = logging.getLogger(__name__)

# 检索结果缓存 TTL（秒）：命中直接返回，文档/库变更由自然过期兜底
_SEARCH_CACHE_TTL = 300


def _to_search_result(doc: dict) -> dict:
    """将 ES 命中文档 _source 映射为 §5.1 检索结果单项"""
    metadata = doc.get("metadata") or {}
    return {
        "chunkId": doc.get("chunk_id"),
        "content": doc.get("content", ""),
        "score": float(doc.get("relevance", 0.0) or 0.0),
        "documentTitle": doc.get("doc_title", ""),
        "documentId": doc.get("doc_id"),
        "chunkIndex": doc.get("chunk_index", 0),
        "metadata": metadata,
    }


def _cosine(a: list[float], b: list[float]) -> float:
    """计算两向量余弦相似度（MMR 候选间多样性用，向量来自 ES _source.content_vector）"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mmr_rerank(candidates: list[dict], top_k: int, lambda_: float = 0.5) -> list[dict]:
    """对候选结果做 MMR 多样性去重，返回按 MMR 分数选择顺序的 top_k 个。

    MMR = λ·relevance(candidate) - (1-λ)·max(cosine(candidate, selected))。
    每次迭代选择当前 MMR 值最高的候选加入结果，故返回顺序为逐轮选择顺序而非原序。
    跨库 embedding 模型异构，relevance 项用候选自身分数（已反映 query 相关度）归一化，
    避免引入单一 query 向量导致异构模型不可比的误差。
    """
    if top_k <= 0 or not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates
    max_score = max(c["relevance"] for c in candidates) or 1.0
    selected: list[dict] = []
    remaining = list(candidates)
    while len(selected) < top_k and remaining:
        best_idx = 0
        best_value = -math.inf
        for i, cand in enumerate(remaining):
            relevance = cand["relevance"] / max_score
            if selected:
                max_sim = max(
                    _cosine(cand["content_vector"], s["content_vector"]) for s in selected
                )
            else:
                max_sim = 0.0
            value = lambda_ * relevance - (1 - lambda_) * max_sim
            if value > best_value:
                best_value = value
                best_idx = i
        selected.append(remaining.pop(best_idx))
    return selected


async def _search_one_kb(
    kb: SysKnowledgeBase,
    query: str,
    query_vector: list[float] | None,
    filters: list[dict],
    top_k: int,
) -> list[dict]:
    """按单个知识库检索策略执行检索并返回 _source 列表（含 content_vector/relevance）"""
    strategy = kb.search_strategy
    if strategy == "vector":
        if not query_vector:
            return []
        docs = await kb_chunk_index.vector_search(
            kb.id, query_vector, filters, top_n=top_k
        )
        # vector 分数为 cosine 相似度(0-1)，低于库配置阈值的结果丢弃
        threshold = float(kb.score_threshold or 0)
        docs = [d for d in docs if d.get("relevance", 0.0) >= threshold]
        return docs
    if strategy == "keyword":
        # BM25 分数无界，不适用 cosine 阈值
        return await kb_chunk_index.keyword_search(kb.id, query, filters, top_n=top_k)
    # hybrid：RRF 融合分数不可比，不适用阈值（启用 rerank 时按 rerank relevance 过滤）
    if not query_vector:
        return []
    return await kb_chunk_index.hybrid_search(
        kb.id,
        query,
        query_vector,
        filters,
        top_k=top_k,
        rank_constant=settings.KB_SEARCH_HYBRID_RANK_CONSTANT,
        rank_window_size=settings.KB_SEARCH_HYBRID_RANK_WINDOW,
        vector_weight=float(kb.hybrid_weight or 1.0),
        keyword_weight=1.0 - float(kb.hybrid_weight or 0.5),
    )


async def _rerank_kb_results(
    kb: SysKnowledgeBase, query: str, docs: list[dict], top_k: int
) -> list[dict]:
    """库启用 rerank 时对 top_k×2 候选重排序取 top_k；失败降级返回原排序（§4.1 上层降级）"""
    if not kb.enable_rerank or not kb.rerank_model or not docs:
        return docs[:top_k]
    documents = [d.get("content", "") for d in docs]
    try:
        ranked = await rerank(kb.embedding_provider, kb.rerank_model, query, documents, top_k)
        # 按 rerank 返回顺序重排；索引无效时回退原序
        order = [r.get("index") for r in ranked]
        if order and all(i is not None for i in order):
            by_index = {i: docs[i] for i in order if 0 <= i < len(docs)}
            return [by_index[i] for i in order if i in by_index]
        return docs[:top_k]
    except Exception as e:  # noqa: BLE001 - rerank 失败降级，不影响主流程
        logger.warning("知识库 %s rerank 失败，降级返回原排序: %s", kb.id, e)
        return docs[:top_k]


async def _retrieve_kb(
    kb: SysKnowledgeBase,
    query: str,
    query_vector: list[float] | None,
    filters: list[dict],
    top_k: int,
) -> list[dict]:
    """单个知识库完整检索链路：检索 + 可选 rerank，返回本库 top_k 结果。"""
    # rerank 需要 top_k×2 候选
    fetch_k = top_k * 2 if kb.enable_rerank and kb.rerank_model else top_k
    docs = await _search_one_kb(kb, query, query_vector, filters, fetch_k)
    return await _rerank_kb_results(kb, query, docs, top_k)


async def _run_retrieval(
    kbs: list[SysKnowledgeBase],
    query: str,
    filters: list[dict],
    top_k: int,
    enable_mmr: bool,
) -> list[dict]:
    """多库检索编排：按 embedding (provider,model) 分组复用 query 向量，组内并发检索。

    各库返回已按本库策略过滤/排序的 top_k 结果，合并后整体截取 top_k；
    enable_mmr 时对合并结果做多样性去重。
    """
    by_model: dict[tuple[str, str], list[SysKnowledgeBase]] = defaultdict(list)
    for kb in kbs:
        by_model[(kb.embedding_provider, kb.embedding_model)].append(kb)

    merged: list[dict] = []
    for (provider, model), group in by_model.items():
        # vector/hybrid 检索需要 query 向量，同模型组一次向量化
        needs_vector = any(kb.search_strategy in ("vector", "hybrid") for kb in group)
        query_vector = (
            await embed_text(provider, model, query) if needs_vector else None
        )
        # 组内多库并发检索，避免多库延迟线性叠加（单库内部已降级返回列表，不会抛异常）
        for docs in await asyncio.gather(
            *(_retrieve_kb(kb, query, query_vector, filters, top_k) for kb in group)
        ):
            merged.extend(docs)

    if enable_mmr:
        merged = _mmr_rerank(merged, top_k)
    else:
        merged = merged[:top_k]
    return merged


def _result_payload(query: str, kb_ids: list[int], docs: list[dict]) -> dict:
    return {
        "query": query,
        "knowledgeBaseIds": kb_ids,
        "results": [_to_search_result(d) for d in docs],
    }


def _cache_key(
    kb_ids: list[int], query: str, filters: list[dict], top_k: int, enable_mmr: bool
) -> str:
    """检索缓存键：库集合哈希 + 查询参数哈希。库集合排序后哈希，顺序无关。"""
    kbs_digest = hashlib.md5(
        json.dumps(sorted(kb_ids), separators=(",", ":")).encode()
    ).hexdigest()[:16]
    params_digest = hashlib.md5(
        json.dumps(
            {"q": query, "f": filters, "k": top_k, "m": enable_mmr},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:16]
    return f"kb:search:{kbs_digest}:{params_digest}"


class SearchService:
    """知识库检索引擎服务（异步）"""

    @staticmethod
    async def search(
        db: AsyncSession,
        redis: Redis,
        user_id: int,
        query: str,
        knowledge_base_ids: list[int] | None = None,
        *,
        top_k: int | None = None,
        filters: dict | None = None,
        enable_mmr: bool = False,
    ) -> dict:
        """多库检索（登录用户）。可见性在 service 层过滤：
        指定库时校验可见性（私有库仅 owner，否则抛 A0301）；未指定时检索该用户全部可见库。
        """
        if knowledge_base_ids is not None and not knowledge_base_ids:
            raise BusinessException(ResultCode.PARAM_ERROR, "knowledgeBaseIds 不能为空数组")

        if knowledge_base_ids:
            kbs = await knowledge_base_repository.get_many(db, knowledge_base_ids)
            found_ids = {kb.id for kb in kbs}
            missing = [i for i in knowledge_base_ids if i not in found_ids]
            if missing:
                raise BusinessException(
                    ResultCode.RESOURCE_NOT_FOUND, f"知识库不存在: {missing}"
                )
            for kb in kbs:
                if kb.visibility == "private" and kb.create_by != user_id:
                    raise BusinessException(
                        ResultCode.ACCESS_UNAUTHORIZED, "无权检索他人私有知识库"
                    )
        else:
            kbs = await knowledge_base_repository.list_visible_by_user(db, user_id)
        if not kbs:
            return {"query": query, "knowledgeBaseIds": [], "results": []}

        return await SearchService._run_search_with_cache(
            redis, kbs, query, top_k, filters, enable_mmr
        )

    @staticmethod
    async def search_internal(
        knowledge_base_id: int | None,
        query: str,
        options: dict | None = None,
    ) -> dict:
        """内部注入检索接口（AI 对话/记忆链路调用），等价单库版 search。

        无用户上下文时仅检索公共库；指定库为非公共库则返回空结果（私有库需显式用户权限）。
        超时/ES 异常降级返回空结果（§6），由 AI 对话降级为无知识回复。
        """
        options = options or {}
        top_k = options.get("topK") or options.get("top_k")
        enable_mmr = options.get("enableMMR") or options.get("enable_mmr") or False
        filters = options.get("filters") or {}

        from app.database import get_db_session
        from app.dependencies.redis import get_redis_client

        async with get_db_session() as db:
            redis = await get_redis_client()
            if knowledge_base_id is not None:
                kb = await knowledge_base_repository.get_by_id(db, knowledge_base_id)
                if not kb or kb.visibility != "public":
                    return {"query": query, "knowledgeBaseIds": [], "results": []}
                kbs = [kb]
            else:
                # 无用户上下文：仅检索公共库
                kbs = await knowledge_base_repository.list_public(db)
            if not kbs:
                return {"query": query, "knowledgeBaseIds": [], "results": []}

            return await SearchService._run_search_with_cache(
                redis, kbs, query, top_k, filters, enable_mmr
            )

    @staticmethod
    async def _run_search_with_cache(
        redis: Redis,
        kbs: list[SysKnowledgeBase],
        query: str,
        top_k: int | None,
        filters: dict | None,
        enable_mmr: bool,
    ) -> dict:
        """检索公共编排：构建参数 → 查缓存 → 检索 → 写缓存 → 构造 payload。

        超时/ES 异常（degraded=True）降级为空结果但**不写缓存**，避免将一次临时降级
        放大为 5 分钟缓存（§6 降级语义）；正常空结果（无匹配）仍可缓存。
        """
        kb_ids = [kb.id for kb in kbs]
        top_k = min(
            top_k or settings.KB_SEARCH_DEFAULT_TOP_K, settings.KB_SEARCH_MAX_TOP_K
        )
        filter_clauses = build_filters(**(filters or {}))
        key = _cache_key(kb_ids, query, filter_clauses, top_k, enable_mmr)

        cached = await redis.get(key)
        if cached:
            return json.loads(cached)

        result, degraded = await SearchService._retrieve_with_timeout(
            kbs, query, filter_clauses, top_k, enable_mmr
        )
        payload = _result_payload(query, kb_ids, result)
        if not degraded:
            await redis.set(key, json.dumps(payload, ensure_ascii=False), ex=_SEARCH_CACHE_TTL)
        return payload

    @staticmethod
    async def _retrieve_with_timeout(
        kbs: list[SysKnowledgeBase],
        query: str,
        filter_clauses: list[dict],
        top_k: int,
        enable_mmr: bool,
    ) -> tuple[list[dict], bool]:
        """整体检索包超时降级：超时/ES 异常返回 (空结果, degraded=True) + 告警日志（§6）。

        返回第二元为是否发生降级，供调用方决定是否写缓存。
        """
        timeout = settings.KB_SEARCH_TIMEOUT_MS / 1000
        try:
            docs = await asyncio.wait_for(
                _run_retrieval(kbs, query, filter_clauses, top_k, enable_mmr),
                timeout=timeout,
            )
            return docs, False
        except TimeoutError:
            logger.warning(
                "知识库检索超时(>%sms)，降级返回空结果 query=%s",
                settings.KB_SEARCH_TIMEOUT_MS,
                query,
            )
            return [], True
        except Exception as e:  # noqa: BLE001 - ES/embedding 异常降级，不影响对话主流程
            logger.warning("知识库检索异常降级返回空结果 query=%s: %s", query, e)
            return [], True


search_service = SearchService()
