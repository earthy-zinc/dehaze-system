from __future__ import annotations

import pytest

pytestmark = pytest.mark.requires_db

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_knowledge_base import SysKnowledgeBase
from app.models.entity.sys_knowledge_chunk import SysKnowledgeChunk
from app.models.entity.sys_knowledge_document import SysKnowledgeDocument
from app.repository.knowledge_chunk_feedback_repository import (
    knowledge_chunk_feedback_repository,
)
from app.service.kb.low_quality_service import LowQualityService


def _make_service():
    # 构造注入仓储：查询聚合用真实仓储，kb 存在性校验用真实仓储
    from app.repository.knowledge_base_repository import knowledge_base_repository

    return LowQualityService(
        knowledge_base_repository=knowledge_base_repository,
        knowledge_chunk_feedback_repository=knowledge_chunk_feedback_repository,
    )


async def _seed_kb_doc_chunk(db, *, kb_id: int, doc_id: int, chunk_ids: list[int]) -> None:
    """预置知识库/文档/分块（知识库为启用状态，避免软删/禁用过滤）。"""
    db.add(
        SysKnowledgeBase(
            id=kb_id,
            name=f"库{kb_id}",
            visibility="private",
            create_by=100,
            status=1,
            embedding_provider="openai",
            embedding_model="bge-m3",
            chunking_strategy="semantic",
            search_strategy="hybrid",
            chunk_size=800,
            chunk_overlap=80,
            top_k=5,
            score_threshold=0.5,
            enable_rerank=0,
            rerank_model=None,
            hybrid_weight=0.5,
            document_count=0,
            chunk_count=0,
            total_tokens=0,
        )
    )
    db.add(
        SysKnowledgeDocument(
            id=doc_id,
            knowledge_base_id=kb_id,
            file_id=None,
            title="文档",
            source="manual",
            version=1,
            parsing_strategy="auto",
            content=None,
            processing_status="completed",
        )
    )
    for i, cid in enumerate(chunk_ids):
        db.add(
            SysKnowledgeChunk(
                id=cid,
                document_id=doc_id,
                knowledge_base_id=kb_id,
                chunk_index=i,
                content=f"片段{i}",
                token_count=1,
            )
        )
    await db.flush()


class TestLowQualityChunks:
    async def test_upsert_feedback_idempotent_counts_once(self, db):
        """同用户对同一片段多次点踩，点踩计数只累加一次（幂等）。"""
        await _seed_kb_doc_chunk(db, kb_id=1, doc_id=1, chunk_ids=[10, 11])
        for _ in range(3):
            await knowledge_chunk_feedback_repository.upsert_feedback(db, 10, 1, -1, None)
        svc = _make_service()
        result = await svc.list_low_quality_chunks(db, 1, 1, 10)
        assert result["total"] == 1
        item = result["list"][0]
        assert item["chunkId"] == 10
        assert item["content"] == "片段0"
        assert item["documentId"] == 1
        assert item["thumbsDownCount"] == 1

    async def test_thumbs_down_counted_desc_by_kb(self, db):
        """多个片段被点踩，按点踩次数降序；不同 kb 数据隔离。"""
        # kb 1：chunk 10 被 2 人点踩、chunk 11 被 1 人点踩
        await _seed_kb_doc_chunk(db, kb_id=1, doc_id=1, chunk_ids=[10, 11])
        # kb 2：chunk 20 也被点踩，不应出现在 kb 1 结果里
        await _seed_kb_doc_chunk(db, kb_id=2, doc_id=2, chunk_ids=[20])
        for chunk_id, users in ((10, (1, 2)), (11, (3,)), (20, (5,))):
            for u in users:
                await knowledge_chunk_feedback_repository.upsert_feedback(
                    db, chunk_id, u, -1, None
                )
        svc = _make_service()
        result = await svc.list_low_quality_chunks(db, 1, 1, 10)
        assert result["total"] == 2
        assert [i["chunkId"] for i in result["list"]] == [10, 11]
        assert result["list"][0]["thumbsDownCount"] == 2
        assert result["list"][1]["thumbsDownCount"] == 1

    async def test_thumbs_up_excluded(self, db):
        """点赞（rating=1）不计入低质量片段。"""
        await _seed_kb_doc_chunk(db, kb_id=1, doc_id=1, chunk_ids=[10, 11])
        await knowledge_chunk_feedback_repository.upsert_feedback(db, 10, 1, -1, None)
        await knowledge_chunk_feedback_repository.upsert_feedback(db, 11, 1, 1, None)
        svc = _make_service()
        result = await svc.list_low_quality_chunks(db, 1, 1, 10)
        assert result["total"] == 1
        assert result["list"][0]["chunkId"] == 10

    async def test_no_feedback_returns_empty(self, db):
        """无任何点踩反馈时返回空列表。"""
        await _seed_kb_doc_chunk(db, kb_id=1, doc_id=1, chunk_ids=[10])
        svc = _make_service()
        result = await svc.list_low_quality_chunks(db, 1, 1, 10)
        assert result == {"list": [], "total": 0}

    async def test_non_existent_kb_raises(self, db):
        """查询不存在知识库的低质量片段抛 A0401。"""
        svc = _make_service()
        with pytest.raises(BusinessException) as excinfo:
            await svc.list_low_quality_chunks(db, 99999999, 1, 10)
        assert excinfo.value.code.code == ResultCode.RESOURCE_NOT_FOUND.code
