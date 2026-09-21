from sqlalchemy import delete, func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_knowledge_chunk import SysKnowledgeChunk
from app.repository.base import BaseRepository


class KnowledgeChunkRepository(BaseRepository[SysKnowledgeChunk]):
    model = SysKnowledgeChunk

    async def list_by_document_sections(
        self, db: AsyncSession, keys: list[tuple[int, int]]
    ) -> list[SysKnowledgeChunk]:
        """按 (document_id, section_index) 节组合拉取全部 child（检索节扩展用）。

        行构造器 IN 精确匹配节组合（document_id/section_index 各自 IN 会拉到
        未命中的交叉组合）；单次查询取回全部节内容，避免逐节查询 N+1。
        """
        if not keys:
            return []
        stmt = (
            select(SysKnowledgeChunk)
            .where(tuple_(SysKnowledgeChunk.document_id, SysKnowledgeChunk.section_index).in_(keys))
            .order_by(SysKnowledgeChunk.document_id.asc(), SysKnowledgeChunk.chunk_index.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def count_by_document(self, db: AsyncSession, document_id: int) -> int:
        """统计文档下的分块数"""
        stmt = (
            select(func.count())
            .select_from(SysKnowledgeChunk)
            .where(SysKnowledgeChunk.document_id == document_id)
        )
        return (await db.execute(stmt)).scalar() or 0

    async def count_by_kb(self, db: AsyncSession, knowledge_base_id: int) -> int:
        """统计知识库下的分块数（单库分块数配额校验用）"""
        stmt = (
            select(func.count())
            .select_from(SysKnowledgeChunk)
            .where(SysKnowledgeChunk.knowledge_base_id == knowledge_base_id)
        )
        return (await db.execute(stmt)).scalar() or 0

    async def delete_by_document(self, db: AsyncSession, document_id: int) -> int:
        """物理删除文档下所有分块（分块只追加，文档更新/删除时整批替换）"""
        stmt = delete(SysKnowledgeChunk).where(SysKnowledgeChunk.document_id == document_id)
        result = await db.execute(stmt)
        return result.rowcount

    async def delete_by_documents(self, db: AsyncSession, document_ids: list[int]) -> int:
        """批量物理删除多个文档下的分块"""
        if not document_ids:
            return 0
        stmt = delete(SysKnowledgeChunk).where(SysKnowledgeChunk.document_id.in_(document_ids))
        result = await db.execute(stmt)
        return result.rowcount


knowledge_chunk_repository = KnowledgeChunkRepository()
