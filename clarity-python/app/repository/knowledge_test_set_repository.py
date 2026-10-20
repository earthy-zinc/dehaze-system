from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_knowledge_test_set import SysKnowledgeTestSet
from app.repository.base import BaseRepository


class KnowledgeTestSetRepository(BaseRepository[SysKnowledgeTestSet]):
    model = SysKnowledgeTestSet

    async def paginate_by_kb(
        self,
        db: AsyncSession,
        knowledge_base_id: int,
        page: int,
        size: int,
    ) -> tuple[list[SysKnowledgeTestSet], int]:
        """按知识库分页查询测试集（未删除，按创建时间倒序）"""
        stmt = (
            select(SysKnowledgeTestSet)
            .where(
                SysKnowledgeTestSet.knowledge_base_id == knowledge_base_id,
                SysKnowledgeTestSet.deleted == 0,
            )
            .order_by(SysKnowledgeTestSet.create_time.desc())
        )
        return await self.paginate(db, stmt, page, size)

    async def get_by_id_and_kb(
        self, db: AsyncSession, test_set_id: int, knowledge_base_id: int
    ) -> SysKnowledgeTestSet | None:
        """按 ID + 知识库查询单个测试集（未删除），用于 run 时归属校验"""
        stmt = select(SysKnowledgeTestSet).where(
            SysKnowledgeTestSet.id == test_set_id,
            SysKnowledgeTestSet.knowledge_base_id == knowledge_base_id,
            SysKnowledgeTestSet.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()


knowledge_test_set_repository = KnowledgeTestSetRepository()
