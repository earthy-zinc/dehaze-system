"""召回测试集服务单元测试：CRUD、Recall@K/hitRate 计算、kb 归属校验。

CRUD/归属用例用真实 db（requires_db），验证落库状态与业务返回值；
run 的召回计算复用 search_service，按该模块的既有模式 patch 底层
knowledge_base_repository / embed_text / kb_chunk_index.vector_search，
构造命中/未命中/期望缺失场景断言 Recall@K 与命中率（只断言业务结果）。
"""

from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_knowledge_base import SysKnowledgeBase
from app.models.entity.sys_knowledge_test_set import SysKnowledgeTestSet
from app.repository.knowledge_test_set_repository import knowledge_test_set_repository
from app.service.kb.test_set_service import test_set_service

CODE_NOT_FOUND = ResultCode.RESOURCE_NOT_FOUND.code

_SS = "app.service.kb.search_service"


def _kb(*, kb_id: int = 1, create_by: int = 100) -> SysKnowledgeBase:
    return SysKnowledgeBase(
        id=kb_id,
        name="测试库",
        description=None,
        visibility="private",
        create_by=create_by,
        status=1,
        embedding_provider="local",
        embedding_model="bge-m3",
        chunking_strategy="fixed",
        search_strategy="vector",
        chunk_size=800,
        chunk_overlap=80,
        top_k=5,
        score_threshold=0.3,
        enable_rerank=0,
        rerank_model=None,
        hybrid_weight=0.5,
        document_count=0,
        chunk_count=0,
        total_tokens=0,
    )


def _retrieval_patches(*docs: dict):
    """模拟单库检索链路：知识库命中 + 向量化 + ES 返回命中文档"""
    kb_repo = AsyncMock()
    kb_repo.get_many.return_value = [_kb()]
    return [
        patch(f"{_SS}.knowledge_base_repository", kb_repo),
        patch(f"{_SS}.embed_text", AsyncMock(return_value=[0.1] * 8)),
        patch(f"{_SS}.kb_chunk_index.vector_search", AsyncMock(return_value=list(docs))),
    ]


def _es_doc(*, chunk_id: int) -> dict:
    return {
        "doc_id": 1,
        "chunk_id": chunk_id,
        "doc_title": "《检索文档》",
        "chunk_index": 0,
        "content": "去雾算法片段",
        "relevance": 0.9,
        "metadata": {},
        "content_vector": [0.1] * 8,
    }


def _enter(patches):
    stack = ExitStack()
    for p in patches:
        stack.enter_context(p)
    return stack


async def _create_and_fetch(db, kb_id: int, question: str, expected: list[int]) -> SysKnowledgeTestSet:
    vo = await test_set_service.create_test_set(db, kb_id, question, expected)
    stmt = select(SysKnowledgeTestSet).where(SysKnowledgeTestSet.id == vo.id)
    return (await db.execute(stmt)).scalar_one()


class TestTestSetCRUD:
    @pytest.mark.requires_db
    async def test_create_test_set_persists_and_returns_vo(self, db):
        vo = await test_set_service.create_test_set(db, 1, "RIDCP 是什么算法？", [10, 11])
        assert vo.id > 0
        assert vo.question == "RIDCP 是什么算法？"
        assert vo.expected_chunk_ids == [10, 11]

        items, total = await knowledge_test_set_repository.paginate_by_kb(db, 1, 1, 10)
        assert total == 1
        assert items[0].question == "RIDCP 是什么算法？"
        assert items[0].expected_chunk_ids == [10, 11]

    @pytest.mark.requires_db
    async def test_list_test_sets_paginated_by_kb(self, db):
        await test_set_service.create_test_set(db, 1, "问题A", [1])
        await test_set_service.create_test_set(db, 1, "问题B", [2])
        await test_set_service.create_test_set(db, 2, "另一库问题", [3])

        result = await test_set_service.list_test_sets(db, 1, 1, 10)
        assert result["total"] == 2
        assert result["list"][0]["id"] > 0
        assert result["list"][0]["question"] in ("问题A", "问题B")
        # 序列化输出 camelCase
        assert "expectedChunkIds" in result["list"][0]
        assert "knowledgeBaseId" in result["list"][0]

    @pytest.mark.requires_db
    async def test_list_test_sets_respects_kb_isolation(self, db):
        await test_set_service.create_test_set(db, 2, "属于库2", [5])
        result = await test_set_service.list_test_sets(db, 1, 1, 10)
        assert result["total"] == 0


class TestTestSetRun:
    @pytest.mark.requires_db
    async def test_run_all_expected_hit_gives_full_recall(self, db, mock_redis):
        ts = await _create_and_fetch(db, 1, "去雾算法", [10, 11])
        with _enter(_retrieval_patches(_es_doc(chunk_id=10), _es_doc(chunk_id=11))):
            result = await test_set_service.run_test_set(
                db, mock_redis, 100, 1, ts.id, top_k=5
            )
        assert result.test_set_id == ts.id
        assert result.recall_at_k == 1.0
        assert result.hit_rate == 1.0
        assert result.total_cases == 1
        assert result.hit_cases == 1

    @pytest.mark.requires_db
    async def test_run_partial_hit_computes_recall_ratio(self, db, mock_redis):
        ts = await _create_and_fetch(db, 1, "去雾算法", [10, 11, 12])
        # 仅召回 chunk 10、11，chunk 12 未命中 → recall 2/3
        with _enter(_retrieval_patches(_es_doc(chunk_id=10), _es_doc(chunk_id=11))):
            result = await test_set_service.run_test_set(
                db, mock_redis, 100, 1, ts.id, top_k=5
            )
        assert result.recall_at_k == pytest.approx(2 / 3)
        assert result.hit_rate == 1.0
        assert result.hit_cases == 1

    @pytest.mark.requires_db
    async def test_run_no_hit_returns_zero_recall_and_zero_hit(self, db, mock_redis):
        ts = await _create_and_fetch(db, 1, "不相关问题", [99])
        # 检索返回空结果（库外问题/低分被阈值过滤）
        with _enter(_retrieval_patches()):
            result = await test_set_service.run_test_set(
                db, mock_redis, 100, 1, ts.id, top_k=5
            )
        assert result.recall_at_k == 0.0
        assert result.hit_rate == 0.0
        assert result.hit_cases == 0

    @pytest.mark.requires_db
    async def test_run_missing_expected_chunk_treated_as_miss(self, db, mock_redis):
        ts = await _create_and_fetch(db, 1, "去雾算法", [999])
        # 期望 chunk 已被删除：检索返回其它 chunk，期望 chunk 缺失按未命中计
        with _enter(_retrieval_patches(_es_doc(chunk_id=1))):
            result = await test_set_service.run_test_set(
                db, mock_redis, 100, 1, ts.id, top_k=5
            )
        assert result.recall_at_k == 0.0
        assert result.hit_cases == 0

    @pytest.mark.requires_db
    async def test_run_search_exception_degrades_to_zero(self, db, mock_redis):
        ts = await _create_and_fetch(db, 1, "去雾算法", [10])
        # 检索异常（ES 宕机）降级为空结果 → 按未命中计，不阻断
        kb_repo = AsyncMock()
        kb_repo.get_many.return_value = [_kb()]
        with _enter(
            [
                patch(f"{_SS}.knowledge_base_repository", kb_repo),
                patch(f"{_SS}.embed_text", AsyncMock(return_value=[0.1] * 8)),
                patch(
                    f"{_SS}.kb_chunk_index.vector_search",
                    AsyncMock(side_effect=RuntimeError("ES 宕机")),
                ),
            ]
        ):
            result = await test_set_service.run_test_set(
                db, mock_redis, 100, 1, ts.id, top_k=5
            )
        assert result.recall_at_k == 0.0
        assert result.hit_cases == 0

    @pytest.mark.requires_db
    async def test_run_empty_expected_gives_zero_recall(self, db, mock_redis):
        # 期望 chunk 列表为空：total_expected 为 0，recall 定义分母为 0 → 计 0.0
        ts = await _create_and_fetch(db, 1, "无期望集问题", [])
        with _enter(_retrieval_patches(_es_doc(chunk_id=1))):
            result = await test_set_service.run_test_set(
                db, mock_redis, 100, 1, ts.id, top_k=5
            )
        assert result.recall_at_k == 0.0
        assert result.hit_rate == 0.0
        assert result.total_cases == 1
        assert result.hit_cases == 0

    @pytest.mark.requires_db
    async def test_run_multi_case_hit_rate_mixed(self, db, mock_redis):
        # 两个用例：用例1全命中（recall 1.0/hit），用例2未命中（recall 0.0/miss）
        ts1 = await _create_and_fetch(db, 1, "去雾算法", [10, 11])
        ts2 = await _create_and_fetch(db, 1, "冷门问题", [999])
        with _enter(
            _retrieval_patches(_es_doc(chunk_id=10), _es_doc(chunk_id=11))
        ):
            r1 = await test_set_service.run_test_set(
                db, mock_redis, 100, 1, ts1.id, top_k=5
            )
            r2 = await test_set_service.run_test_set(
                db, mock_redis, 100, 1, ts2.id, top_k=5
            )
        assert r1.recall_at_k == 1.0 and r1.hit_cases == 1
        assert r2.recall_at_k == 0.0 and r2.hit_cases == 0
        # 跨用例聚合口径：命中率 = 命中用例数 / 总用例数
        agg_hit_rate = (r1.hit_cases + r2.hit_cases) / (r1.total_cases + r2.total_cases)
        assert agg_hit_rate == pytest.approx(0.5)


class TestTestSetOwnership:
    @pytest.mark.requires_db
    async def test_run_test_set_of_other_kb_raises_not_found(self, db, mock_redis):
        ts = await _create_and_fetch(db, 2, "库2问题", [1])
        # 在库1名下执行库2的测试集 → 归属校验失败（A0401）
        with pytest.raises(BusinessException) as excinfo:
            await test_set_service.run_test_set(db, mock_redis, 100, 1, ts.id, top_k=5)
        assert excinfo.value.code.code == CODE_NOT_FOUND

    @pytest.mark.requires_db
    async def test_run_nonexistent_test_set_raises_not_found(self, db, mock_redis):
        with pytest.raises(BusinessException) as excinfo:
            await test_set_service.run_test_set(db, mock_redis, 100, 1, 99999, top_k=5)
        assert excinfo.value.code.code == CODE_NOT_FOUND
