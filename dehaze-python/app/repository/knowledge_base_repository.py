from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_knowledge_base import SysKnowledgeBase
from app.repository.base import BaseRepository, escape_like


class KnowledgeBaseRepository(BaseRepository[SysKnowledgeBase]):
    model = SysKnowledgeBase

    async def get_by_id(self, db: AsyncSession, id: int) -> SysKnowledgeBase | None:
        """按 ID 查询（默认过滤软删）"""
        stmt = select(SysKnowledgeBase).where(SysKnowledgeBase.id == id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id_include_deleted(
        self, db: AsyncSession, id: int
    ) -> SysKnowledgeBase | None:
        """按 ID 查询（含软删，用于同名校验绕过软删过滤）"""
        stmt = select(SysKnowledgeBase).where(SysKnowledgeBase.id == id)
        stmt = stmt.execution_options(include_deleted=True)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_name_and_owner(
        self,
        db: AsyncSession,
        name: str,
        create_by: int,
    ) -> SysKnowledgeBase | None:
        """同名校验：同 create_by 且未删除的知识库"""
        stmt = select(SysKnowledgeBase).where(
            SysKnowledgeBase.name == name,
            SysKnowledgeBase.create_by == create_by,
            SysKnowledgeBase.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def count_private_by_owner(self, db: AsyncSession, create_by: int) -> int:
        """统计创建者的私有库数量（创建私有库时配额校验用）"""
        stmt = select(func.count()).select_from(SysKnowledgeBase).where(
            SysKnowledgeBase.create_by == create_by,
            SysKnowledgeBase.visibility == "private",
            SysKnowledgeBase.deleted == 0,
        )
        return (await db.execute(stmt)).scalar() or 0

    async def paginate_visible(
        self,
        db: AsyncSession,
        user_id: int,
        keyword: str | None,
        page: int,
        size: int,
    ) -> tuple[list[SysKnowledgeBase], int]:
        """分页查询当前用户可见的知识库（私有库仅本人 + 公共库全员）"""
        stmt = select(SysKnowledgeBase).where(
            (SysKnowledgeBase.visibility == "public")
            | (
                (SysKnowledgeBase.visibility == "private")
                & (SysKnowledgeBase.create_by == user_id)
            ),
            SysKnowledgeBase.deleted == 0,
        )
        if keyword:
            stmt = stmt.where(
                SysKnowledgeBase.name.like(f"%{escape_like(keyword)}%", escape="\\")
            )
        stmt = stmt.order_by(SysKnowledgeBase.create_time.desc())
        return await self.paginate(db, stmt, page, size)

    async def list_visible_by_user(self, db: AsyncSession, user_id: int) -> list[SysKnowledgeBase]:
        """查询当前用户全部可见知识库（私有库仅本人 + 公共库全员），用于缺省多库检索。"""
        stmt = (
            select(SysKnowledgeBase)
            .where(
                (SysKnowledgeBase.visibility == "public")
                | (
                    (SysKnowledgeBase.visibility == "private")
                    & (SysKnowledgeBase.create_by == user_id)
                ),
                SysKnowledgeBase.deleted == 0,
                SysKnowledgeBase.status == 1,
            )
            .order_by(SysKnowledgeBase.create_time.desc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def list_public(self, db: AsyncSession) -> list[SysKnowledgeBase]:
        """查询全部公共知识库（无用户上下文内部检索用）"""
        stmt = (
            select(SysKnowledgeBase)
            .where(
                SysKnowledgeBase.visibility == "public",
                SysKnowledgeBase.deleted == 0,
                SysKnowledgeBase.status == 1,
            )
            .order_by(SysKnowledgeBase.create_time.desc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_many(self, db: AsyncSession, ids: list[int]) -> list[SysKnowledgeBase]:
        """按 ID 批量查询（默认过滤软删与禁用库）"""
        stmt = select(SysKnowledgeBase).where(
            SysKnowledgeBase.id.in_(ids),
            SysKnowledgeBase.deleted == 0,
            SysKnowledgeBase.status == 1,
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def update_stats_cas(
        self,
        db: AsyncSession,
        kb_id: int,
        document_delta: int,
        chunk_delta: int,
        token_delta: int,
    ) -> bool:
        """统计字段原子更新（乐观锁 CAS，携带原值快照，冲突返回 False 由调用方重试）。

        文档处理完成时一次性累加三个冗余统计字段，避免并发覆盖。
        """
        kb = await self.get_by_id(db, kb_id)
        if not kb:
            return False
        stmt = (
            update(SysKnowledgeBase)
            .where(
                SysKnowledgeBase.id == kb_id,
                SysKnowledgeBase.document_count == kb.document_count,
                SysKnowledgeBase.chunk_count == kb.chunk_count,
                SysKnowledgeBase.total_tokens == kb.total_tokens,
            )
            .values(
                document_count=SysKnowledgeBase.document_count + document_delta,
                chunk_count=SysKnowledgeBase.chunk_count + chunk_delta,
                total_tokens=SysKnowledgeBase.total_tokens + token_delta,
            )
        )
        result = await db.execute(stmt)
        return result.rowcount == 1


knowledge_base_repository = KnowledgeBaseRepository()
