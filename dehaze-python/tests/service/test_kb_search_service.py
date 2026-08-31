import asyncio
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.es import kb_chunk_index
from app.models.entity.sys_knowledge_base import SysKnowledgeBase
from app.service.kb.search_service import search_service, _mmr_rerank

CODE_UNAUTHORIZED = ResultCode.ACCESS_UNAUTHORIZED.code
CODE_NOT_FOUND = ResultCode.RESOURCE_NOT_FOUND.code

_SI = "app.service.kb.search_service"


def _kb(
    *,
    kb_id: int = 1,
    visibility: str = "public",
    create_by: int = 100,
    search_strategy: str = "vector",
    score_threshold: float = 0.0,
    enable_rerank: int = 0,
    rerank_model: str | None = None,
    hybrid_weight: float = 0.5,
) -> SysKnowledgeBase:
    return SysKnowledgeBase(
        id=kb_id,
        name=f"kb{kb_id}",
        description=None,
        visibility=visibility,
        create_by=create_by,
        status=1,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        chunking_strategy="semantic",
        search_strategy=search_strategy,
        chunk_size=800,
        chunk_overlap=80,
        top_k=5,
        score_threshold=score_threshold,
        enable_rerank=enable_rerank,
        rerank_model=rerank_model,
        hybrid_weight=hybrid_weight,
        document_count=0,
        chunk_count=0,
        total_tokens=0,
    )


def _es_doc(*, doc_id: int, chunk_id: int, content: str, relevance: float, idx: int = 0):
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "doc_title": f"《检索文档{doc_id}》",
        "chunk_index": idx,
        "content": content,
        "relevance": relevance,
        "metadata": {"source": "test"},
        "content_vector": [0.1] * 8,
    }


def _enter(patches):
    stack = ExitStack()
    for p in patches:
        stack.enter_context(p)
    return stack


def _vec_patch(*docs):
    return patch(f"{_SI}.kb_chunk_index.vector_search", AsyncMock(return_value=list(docs)))


def _embed_mock(vector=None):
    return AsyncMock(return_value=vector or [0.1] * 8)


def _fake_settings(**over):
    base = dict(
        KB_SEARCH_TIMEOUT_MS=5000,
        KB_SEARCH_DEFAULT_TOP_K=5,
        KB_SEARCH_MAX_TOP_K=20,
        KB_SEARCH_HYBRID_RANK_CONSTANT=60,
        KB_SEARCH_HYBRID_RANK_WINDOW=20,
    )
    base.update(over)
    return SimpleNamespace(**base)


class TestSearchVisibility:
    async def test_private_kb_of_others_raises(self, mock_redis):
        kb_repo = AsyncMock()
        kb_repo.get_many.return_value = [_kb(kb_id=1, visibility="private", create_by=200)]
        with patch(f"{_SI}.knowledge_base_repository", kb_repo):
            with pytest.raises(BusinessException) as excinfo:
                await search_service.search(
                    None, mock_redis, 100, "量子纠缠科普", knowledge_base_ids=[1]
                )
            assert excinfo.value.code.code == CODE_UNAUTHORIZED

    async def test_public_kb_allowed(self, mock_redis):
        kb = _kb(kb_id=1, visibility="public")
        kb_repo = AsyncMock()
        kb_repo.get_many.return_value = [kb]
        content = "量子纠缠的观测坍缩，遵循全角标点；half-width mixed."
        with _enter(
            [
                patch(f"{_SI}.knowledge_base_repository", kb_repo),
                patch(f"{_SI}.embed_text", _embed_mock()),
                _vec_patch(_es_doc(doc_id=1, chunk_id=1, content=content, relevance=0.9)),
            ]
        ):
            result = await search_service.search(
                None, mock_redis, 100, "量子纠缠", knowledge_base_ids=[1]
            )
        assert len(result["results"]) == 1
        assert result["results"][0]["content"] == content
        assert result["results"][0]["score"] == pytest.approx(0.9)

    async def test_missing_kb_raises_not_found(self, mock_redis):
        kb_repo = AsyncMock()
        kb_repo.get_many.return_value = []
        with patch(f"{_SI}.knowledge_base_repository", kb_repo):
            with pytest.raises(BusinessException) as excinfo:
                await search_service.search(
                    None, mock_redis, 100, "缺失知识库", knowledge_base_ids=[999]
                )
            assert excinfo.value.code.code == CODE_NOT_FOUND

    async def test_empty_kb_list_returns_empty_result(self, mock_redis):
        kb_repo = AsyncMock()
        kb_repo.list_visible_by_user.return_value = []
        with patch(f"{_SI}.knowledge_base_repository", kb_repo):
            result = await search_service.search(None, mock_redis, 100, "无可见库检索")
        assert result["results"] == []
        assert result["knowledgeBaseIds"] == []

    async def test_empty_ids_array_raises_param_error(self, mock_redis):
        with pytest.raises(BusinessException) as excinfo:
            await search_service.search(None, mock_redis, 100, "空数组", knowledge_base_ids=[])
        assert excinfo.value.code.code == ResultCode.PARAM_ERROR.code


class TestSearchThreshold:
    async def test_below_threshold_filtered_out(self, mock_redis):
        kb = _kb(kb_id=1, search_strategy="vector", score_threshold=0.8)
        kb_repo = AsyncMock()
        kb_repo.get_many.return_value = [kb]
        with _enter(
            [
                patch(f"{_SI}.knowledge_base_repository", kb_repo),
                patch(f"{_SI}.embed_text", _embed_mock()),
                _vec_patch(
                    _es_doc(doc_id=1, chunk_id=1, content="语义相关度偏低的片段", relevance=0.3)
                ),
            ]
        ):
            result = await search_service.search(
                None, mock_redis, 100, "阈值过滤", knowledge_base_ids=[1]
            )
        assert result["results"] == []

    async def test_at_or_above_threshold_kept(self, mock_redis):
        kb = _kb(kb_id=1, search_strategy="vector", score_threshold=0.5)
        kb_repo = AsyncMock()
        kb_repo.get_many.return_value = [kb]
        with _enter(
            [
                patch(f"{_SI}.knowledge_base_repository", kb_repo),
                patch(f"{_SI}.embed_text", _embed_mock()),
                _vec_patch(
                    _es_doc(doc_id=1, chunk_id=1, content="超过阈值保留的片段", relevance=0.7)
                ),
            ]
        ):
            result = await search_service.search(
                None, mock_redis, 100, "阈值过滤", knowledge_base_ids=[1]
            )
        assert len(result["results"]) == 1
        assert result["results"][0]["score"] == pytest.approx(0.7)


class TestSearchRerank:
    def _base_patches(self, docs):
        kb = _kb(kb_id=1, enable_rerank=1, rerank_model="rerank-1", search_strategy="keyword")
        kb_repo = AsyncMock()
        kb_repo.get_many.return_value = [kb]
        return [
            patch(f"{_SI}.knowledge_base_repository", kb_repo),
            patch(f"{_SI}.kb_chunk_index.keyword_search", AsyncMock(return_value=docs)),
        ]

    async def test_rerank_reorders_by_index(self, mock_redis):
        docs = [
            _es_doc(doc_id=1, chunk_id=1, content="候选一：全文检索命中段落", relevance=0.9),
            _es_doc(doc_id=2, chunk_id=2, content="候选二：语义重排优先段落", relevance=0.8),
        ]
        patches = self._base_patches(docs)
        rerank_mock = AsyncMock(return_value=[{"index": 1}, {"index": 0}])
        patches.append(patch(f"{_SI}.rerank", rerank_mock))
        with _enter(patches):
            result = await search_service.search(
                None, mock_redis, 100, "检索重排", knowledge_base_ids=[1], top_k=2
            )
        contents = [r["content"] for r in result["results"]]
        assert contents == ["候选二：语义重排优先段落", "候选一：全文检索命中段落"]

    async def test_rerank_failure_degrades_to_original_order(self, mock_redis):
        docs = [
            _es_doc(doc_id=1, chunk_id=1, content="候选一：全文检索命中段落", relevance=0.9),
            _es_doc(doc_id=2, chunk_id=2, content="候选二：语义重排优先段落", relevance=0.8),
        ]
        patches = self._base_patches(docs)
        rerank_mock = AsyncMock(side_effect=RuntimeError("rerank 宕机"))
        patches.append(patch(f"{_SI}.rerank", rerank_mock))
        with _enter(patches):
            result = await search_service.search(
                None, mock_redis, 100, "检索重排", knowledge_base_ids=[1], top_k=2
            )
        contents = [r["content"] for r in result["results"]]
        assert contents == ["候选一：全文检索命中段落", "候选二：语义重排优先段落"]


class TestSearchTimeout:
    async def test_timeout_returns_empty_no_cache(self, mock_redis):
        kb = _kb(kb_id=1)
        kb_repo = AsyncMock()
        kb_repo.get_many.return_value = [kb]

        async def _slow_embed(provider, model, text):
            await asyncio.sleep(1)
            return [0.1] * 8

        with _enter(
            [
                patch(f"{_SI}.knowledge_base_repository", kb_repo),
                patch(f"{_SI}.embed_text", _slow_embed),
                patch(f"{_SI}.settings", _fake_settings(KB_SEARCH_TIMEOUT_MS=1)),
                _vec_patch(),
            ]
        ):
            result = await search_service.search(
                None, mock_redis, 100, "超时降级", knowledge_base_ids=[1]
            )
        assert result["results"] == []
        assert await mock_redis.keys("kb:search:*") == []


class TestSearchDegraded:
    async def test_es_all_raise_returns_empty(self, mock_redis):
        kb = _kb(kb_id=1)
        kb_repo = AsyncMock()
        kb_repo.get_many.return_value = [kb]
        es_err = AsyncMock(side_effect=RuntimeError("ES 宕机"))
        with _enter(
            [
                patch(f"{_SI}.knowledge_base_repository", kb_repo),
                patch(f"{_SI}.embed_text", _embed_mock()),
                patch(f"{_SI}.kb_chunk_index.vector_search", es_err),
            ]
        ):
            result = await search_service.search(
                None, mock_redis, 100, "ES 异常检索", knowledge_base_ids=[1]
            )
        assert result["results"] == []
        assert await mock_redis.keys("kb:search:*") == []


class TestSearchCache:
    async def test_second_call_hits_cache_no_retrieval(self, mock_redis):
        kb = _kb(kb_id=1)
        kb_repo = AsyncMock()
        kb_repo.get_many.return_value = [kb]
        embed_mock = _embed_mock()
        content = "缓存命中片段：量子纠错的容错阈值"
        with _enter(
            [
                patch(f"{_SI}.knowledge_base_repository", kb_repo),
                patch(f"{_SI}.embed_text", embed_mock),
                _vec_patch(_es_doc(doc_id=1, chunk_id=1, content=content, relevance=0.9)),
            ]
        ):
            first = await search_service.search(
                None, mock_redis, 100, "缓存验证", knowledge_base_ids=[1]
            )
            second = await search_service.search(
                None, mock_redis, 100, "缓存验证", knowledge_base_ids=[1]
            )
        assert embed_mock.call_count == 1
        assert second == first
        assert second["results"][0]["content"] == content

    async def test_top_k_change_misses_cache_and_retrieves(self, mock_redis):
        kb = _kb(kb_id=1)
        kb_repo = AsyncMock()
        kb_repo.get_many.return_value = [kb]
        embed_mock = _embed_mock()
        with _enter(
            [
                patch(f"{_SI}.knowledge_base_repository", kb_repo),
                patch(f"{_SI}.embed_text", embed_mock),
                _vec_patch(_es_doc(doc_id=1, chunk_id=1, content="缓存键随参数变化", relevance=0.9)),
            ]
        ):
            await search_service.search(
                None, mock_redis, 100, "缓存失效", knowledge_base_ids=[1], top_k=3
            )
            await search_service.search(
                None, mock_redis, 100, "缓存失效", knowledge_base_ids=[1], top_k=7
            )
        assert embed_mock.call_count == 2


class TestMmrRerank:
    @staticmethod
    def _cand(cid: int, relevance: float, vector: list[float]) -> dict:
        return {"chunk_id": cid, "relevance": relevance, "content_vector": vector}

    def test_duplicate_suppressed_for_diversity(self):
        a = self._cand(1, 1.0, [1.0, 0.0])
        b = self._cand(2, 0.95, [1.0, 0.0])
        c = self._cand(3, 0.8, [0.0, 1.0])
        d = self._cand(4, 0.75, [0.0, -1.0])
        assert _mmr_rerank([a, b, c, d], top_k=3) == [a, c, d]

    def test_selection_order_by_mmr_value(self):
        cands = [self._cand(i, 1.0 - i * 0.1, [float(i), 1.0]) for i in range(4)]
        out = _mmr_rerank(cands, top_k=2)
        assert [c["chunk_id"] for c in out] == [0, 3]

    def test_fewer_candidates_than_top_k_passthrough(self):
        cands = [self._cand(1, 0.5, [1.0]), self._cand(2, 0.4, [0.0])]
        assert _mmr_rerank(cands, top_k=5) == cands

    def test_zero_top_k_returns_empty(self):
        assert _mmr_rerank([self._cand(1, 1.0, [1.0])], top_k=0) == []

    def test_empty_candidates_returns_empty(self):
        assert _mmr_rerank([], top_k=3) == []


_KI = "app.infrastructure.es.kb_chunk_index"


def _hybrid_doc(chunk_id: int) -> dict:
    doc = _es_doc(
        doc_id=chunk_id, chunk_id=chunk_id, content=f"片段{chunk_id}", relevance=0.9
    )
    doc.pop("content_vector")  # RRF 融合只依赖排名，keyword 候选无向量
    return doc


class TestHybridRrfFusion:
    """应用层 RRF 融合（kb_chunk_index.hybrid_search）：vector+keyword 候选按排名融合"""

    VECTOR_DOCS = [_hybrid_doc(1), _hybrid_doc(2), _hybrid_doc(3)]
    KEYWORD_DOCS = [_hybrid_doc(2), _hybrid_doc(4), _hybrid_doc(1)]

    @staticmethod
    def _patches(vector_docs, keyword_docs):
        return [
            patch(f"{_KI}.vector_search", AsyncMock(return_value=list(vector_docs))),
            patch(f"{_KI}.keyword_search", AsyncMock(return_value=list(keyword_docs))),
        ]

    async def test_fusion_orders_by_rank_not_raw_score(self):
        # 语义：B 双榜命中排名高 → 融合第一；原始 relevance 分不参与融合
        with _enter(self._patches(self.VECTOR_DOCS, self.KEYWORD_DOCS)):
            docs = await kb_chunk_index.hybrid_search(1, "q", [0.1] * 8, [], top_k=5)
        assert [d["chunk_id"] for d in docs] == [2, 1, 4, 3]
        # score = 1/(60+rank)：B=1/61+1/62，A=1/61+1/63
        assert docs[0]["relevance"] == pytest.approx(1 / 61 + 1 / 62)
        assert docs[1]["relevance"] == pytest.approx(1 / 61 + 1 / 63)

    async def test_vector_weight_breaks_tie_toward_vector_ranking(self):
        # 语义：大权重下拉高 vector 榜排名的相对位次（A 反超 B），keyword-only 命中 D 沉底
        with _enter(self._patches(self.VECTOR_DOCS, self.KEYWORD_DOCS)):
            docs = await kb_chunk_index.hybrid_search(
                1, "q", [0.1] * 8, [], top_k=5, vector_weight=10.0
            )
        assert [d["chunk_id"] for d in docs] == [1, 2, 3, 4]

    async def test_top_k_truncates_fused_results(self):
        with _enter(self._patches(self.VECTOR_DOCS, self.KEYWORD_DOCS)):
            docs = await kb_chunk_index.hybrid_search(1, "q", [0.1] * 8, [], top_k=2)
        assert [d["chunk_id"] for d in docs] == [2, 1]

    async def test_empty_query_vector_returns_empty_without_search(self):
        vec_mock, kw_mock = AsyncMock(), AsyncMock()
        with _enter(
            [
                patch(f"{_KI}.vector_search", vec_mock),
                patch(f"{_KI}.keyword_search", kw_mock),
            ]
        ):
            docs = await kb_chunk_index.hybrid_search(1, "q", [], [], top_k=5)
        assert docs == []
        vec_mock.assert_not_called()
        kw_mock.assert_not_called()

    async def test_filters_passed_to_both_sub_searches(self):
        vec_mock = AsyncMock(return_value=list(self.VECTOR_DOCS))
        kw_mock = AsyncMock(return_value=list(self.KEYWORD_DOCS))
        filters = [{"terms": {"tags": ["x"]}}]
        with _enter(
            [
                patch(f"{_KI}.vector_search", vec_mock),
                patch(f"{_KI}.keyword_search", kw_mock),
            ]
        ):
            await kb_chunk_index.hybrid_search(1, "q", [0.1] * 8, filters, top_k=5)
        assert vec_mock.await_args.args[2] == filters
        assert kw_mock.await_args.args[2] == filters


class TestSearchHybridStrategy:
    """search_service hybrid 策略分发：权重透传与空向量短路"""

    def _patches(self, vector_docs, keyword_docs):
        kb = _kb(kb_id=1, search_strategy="hybrid", hybrid_weight=0.7)
        kb_repo = AsyncMock()
        kb_repo.get_many.return_value = [kb]
        return [
            patch(f"{_SI}.knowledge_base_repository", kb_repo),
            patch(f"{_SI}.embed_text", _embed_mock()),
            patch(f"{_KI}.vector_search", AsyncMock(return_value=vector_docs)),
            patch(f"{_KI}.keyword_search", AsyncMock(return_value=keyword_docs)),
        ]

    async def test_hybrid_dispatch_uses_rrf_fused_order(self, mock_redis):
        vector_docs = [_hybrid_doc(1), _hybrid_doc(2)]
        keyword_docs = [_hybrid_doc(2), _hybrid_doc(3)]
        with _enter(self._patches(vector_docs, keyword_docs)):
            result = await search_service.search(
                None, mock_redis, 100, "混合检索", knowledge_base_ids=[1]
            )
        # hybrid_weight=0.7 → vector_weight=0.7/keyword_weight=0.3，
        # 融合分 2: 0.7/61+0.3/61=0.01639 > 1: 0.7/62+0.3/62=0.01613
        assert [r["chunkId"] for r in result["results"]] == [2, 1, 3]

    async def test_hybrid_empty_query_vector_returns_empty(self, mock_redis):
        kb = _kb(kb_id=1, search_strategy="hybrid")
        kb_repo = AsyncMock()
        kb_repo.get_many.return_value = [kb]
        vec_mock = AsyncMock()
        with _enter(
            [
                patch(f"{_SI}.knowledge_base_repository", kb_repo),
                patch(f"{_SI}.embed_text", AsyncMock(return_value=None)),
                patch(f"{_KI}.vector_search", vec_mock),
            ]
        ):
            result = await search_service.search(
                None, mock_redis, 100, "空向量混合", knowledge_base_ids=[1]
            )
        assert result["results"] == []
        vec_mock.assert_not_called()
