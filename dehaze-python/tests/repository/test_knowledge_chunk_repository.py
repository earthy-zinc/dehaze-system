"""knowledge_chunk_repository 按节组合拉取 child 的真实 SQL 语义测试（MySQL dehaze_test）"""

import pytest

from app.models.entity.sys_knowledge_chunk import SysKnowledgeChunk
from app.repository.knowledge_chunk_repository import knowledge_chunk_repository

pytestmark = pytest.mark.requires_db


async def _make_chunk(
    db,
    *,
    document_id: int,
    chunk_index: int,
    section_index: int = 0,
    content: str = "内容",
    section_path: str | None = None,
    kb_id: int = 1,
) -> SysKnowledgeChunk:
    chunk = SysKnowledgeChunk(
        document_id=document_id,
        knowledge_base_id=kb_id,
        chunk_index=chunk_index,
        section_index=section_index,
        section_path=section_path,
        content=content,
        token_count=10,
    )
    db.add(chunk)
    await db.flush()
    return chunk


async def test_list_by_document_sections_orders_by_chunk_index(db):
    """节内容拼接依赖 SQL 按 chunk_index 升序（插入乱序验证排序语义）"""
    second = await _make_chunk(db, document_id=1, chunk_index=1, content="乙段")
    first = await _make_chunk(db, document_id=1, chunk_index=0, content="甲段")
    rows = await knowledge_chunk_repository.list_by_document_sections(db, [(1, 0)])
    assert [r.id for r in rows] == [first.id, second.id]
    assert [r.content for r in rows] == ["甲段", "乙段"]


async def test_list_by_document_sections_exact_key_match(db):
    """行构造器 IN 精确匹配 (document_id, section_index)：交叉组合不误拉"""
    unrelated = await _make_chunk(
        db, document_id=1, chunk_index=0, section_index=9, content="未命中节"
    )
    hit = await _make_chunk(db, document_id=1, chunk_index=0, section_index=2, content="命中节")
    other_doc = await _make_chunk(
        db, document_id=3, chunk_index=0, section_index=2, content="其他文档同节"
    )
    rows = await knowledge_chunk_repository.list_by_document_sections(db, [(1, 2), (3, 2)])
    ids = {r.id for r in rows}
    assert ids == {hit.id, other_doc.id}
    assert unrelated.id not in ids


async def test_list_by_document_sections_empty_keys_returns_empty(db):
    assert await knowledge_chunk_repository.list_by_document_sections(db, []) == []


async def test_list_by_document_sections_covers_section_path(db):
    """section_path 随行返回：检索节扩展零回表拿到注入锚点"""
    await _make_chunk(
        db,
        document_id=1,
        chunk_index=0,
        section_index=4,
        content="节内容",
        section_path="4 核心设计 > 4.2 检索引擎",
    )
    rows = await knowledge_chunk_repository.list_by_document_sections(db, [(1, 4)])
    assert rows[0].section_path == "4 核心设计 > 4.2 检索引擎"
