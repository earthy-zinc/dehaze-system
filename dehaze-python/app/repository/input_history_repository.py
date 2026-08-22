"""
图像输入历史记录 Repository
对齐 dehaze-java SysInputHistory 字段
"""

from sqlalchemy import delete, desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_input_history import SysInputHistory
from app.repository.base import BaseRepository, escape_like


class InputHistoryRepository(BaseRepository[SysInputHistory]):
    """图像输入历史记录 Repository"""

    model = SysInputHistory

    async def get_paginated(
        self,
        db: AsyncSession,
        user_id: int,
        status: int | None = None,
        input_source: str | None = None,
        keywords: str | None = None,
        page: int = 1,
        size: int = 10,
    ) -> tuple[list[SysInputHistory], int]:
        """分页查询历史记录（按用户隔离）"""
        stmt = select(SysInputHistory).where(SysInputHistory.user_id == user_id)
        if status is not None:
            stmt = stmt.where(SysInputHistory.status == status)
        if input_source:
            stmt = stmt.where(SysInputHistory.input_source == input_source)
        if keywords:
            stmt = stmt.where(
                or_(
                    SysInputHistory.algorithm_name.like(f"%{escape_like(keywords)}%", escape="\\"),
                    SysInputHistory.original_image_url.like(
                        f"%{escape_like(keywords)}%", escape="\\"
                    ),
                )
            )
        stmt = stmt.order_by(desc(SysInputHistory.id))
        return await self.paginate(db, stmt, page, size)

    async def create_history(self, db: AsyncSession, **kwargs) -> SysInputHistory:
        """创建历史记录"""
        history = SysInputHistory(**kwargs)
        return await self.create(db, history)

    async def delete_by_user(self, db: AsyncSession, user_id: int, history_id: int) -> bool:
        """删除单条（仅限本人）"""
        stmt = (
            delete(SysInputHistory)
            .where(SysInputHistory.id == history_id)
            .where(SysInputHistory.user_id == user_id)
        )
        result = await db.execute(stmt)
        return result.rowcount > 0

    async def batch_delete_by_user(
        self,
        db: AsyncSession,
        user_id: int,
        ids: list[int],
    ) -> int:
        """批量删除（仅限本人）"""
        if not ids:
            return 0
        stmt = (
            delete(SysInputHistory)
            .where(SysInputHistory.id.in_(ids))
            .where(SysInputHistory.user_id == user_id)
        )
        result = await db.execute(stmt)
        return result.rowcount

    async def clear_by_user(self, db: AsyncSession, user_id: int) -> int:
        """清空用户所有历史记录"""
        stmt = delete(SysInputHistory).where(SysInputHistory.user_id == user_id)
        result = await db.execute(stmt)
        return result.rowcount


input_history_repository = InputHistoryRepository()
