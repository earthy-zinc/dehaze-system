from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_knowledge_document import SysKnowledgeDocument
from app.repository.base import BaseRepository


class KnowledgeDocumentRepository(BaseRepository[SysKnowledgeDocument]):
    model = SysKnowledgeDocument

    async def paginate_by_kb(
        self,
        db: AsyncSession,
        knowledge_base_id: int,
        processing_status: str | None,
        page: int,
        size: int,
    ) -> tuple[list[SysKnowledgeDocument], int]:
        """按知识库分页查询文档列表（未删除）"""
        stmt = select(SysKnowledgeDocument).where(
            SysKnowledgeDocument.knowledge_base_id == knowledge_base_id,
            SysKnowledgeDocument.deleted == 0,
        )
        if processing_status:
            stmt = stmt.where(SysKnowledgeDocument.processing_status == processing_status)
        stmt = stmt.order_by(SysKnowledgeDocument.create_time.desc())
        return await self.paginate(db, stmt, page, size)

    async def get_by_file_id(
        self,
        db: AsyncSession,
        knowledge_base_id: int,
        file_id: int,
    ) -> SysKnowledgeDocument | None:
        """按 file_id 查重：同库同 file_id 已存在且未删除的文档（批量上传幂等去重）"""
        stmt = select(SysKnowledgeDocument).where(
            SysKnowledgeDocument.knowledge_base_id == knowledge_base_id,
            SysKnowledgeDocument.file_id == file_id,
            SysKnowledgeDocument.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def count_by_kb(
        self, db: AsyncSession, knowledge_base_id: int
    ) -> int:
        """统计知识库下未删除的文档数（单库文档数配额校验用）"""
        stmt = select(func.count()).select_from(SysKnowledgeDocument).where(
            SysKnowledgeDocument.knowledge_base_id == knowledge_base_id,
            SysKnowledgeDocument.deleted == 0,
        )
        return (await db.execute(stmt)).scalar() or 0

    async def list_ids_by_kb(self, db: AsyncSession, knowledge_base_id: int) -> list[int]:
        """查询知识库下未删除文档的 ID 列表（删除知识库时软删文档用）"""
        stmt = select(SysKnowledgeDocument.id).where(
            SysKnowledgeDocument.knowledge_base_id == knowledge_base_id,
            SysKnowledgeDocument.deleted == 0,
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())


knowledge_document_repository = KnowledgeDocumentRepository()
