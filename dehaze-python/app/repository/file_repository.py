"""
文件数据访问层
"""

from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_file import SysFile
from app.repository.base import BaseRepository


class FileRepository(BaseRepository[SysFile]):
    """文件数据访问层"""

    model = SysFile

    async def get_by_md5(
        self,
        db: AsyncSession,
        md5: str,
    ) -> SysFile | None:
        """根据 MD5 查询文件（去重用，仅查未删除记录）"""
        stmt = select(SysFile).where(SysFile.md5 == md5, SysFile.deleted == 0)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def upsert_by_md5(
        self,
        db: AsyncSession,
        *,
        md5: str,
        type: str,
        name: str,
        object_name: str,
        storage: str,
        size: str,
        size_bytes: int,
    ) -> SysFile:
        """upsert 文件记录：冲突时复活（重置 deleted=0），返回记录"""
        stmt = mysql_insert(SysFile).values(
            md5=md5,
            type=type,
            name=name,
            object_name=object_name,
            storage=storage,
            size=size,
            size_bytes=size_bytes,
        )
        stmt = stmt.on_duplicate_key_update(
            type=type,
            name=name,
            object_name=object_name,
            storage=storage,
            size=size,
            size_bytes=size_bytes,
            deleted=0,
            update_time=func.now(),
        )
        await db.execute(stmt)
        # 用当前读（FOR UPDATE）确保看到 ODKU 的结果，避免 RR 隔离级别下
        # 并发场景快照读看不到刚 upsert 的行（autoflush=False 时尤为关键）
        result = await db.execute(
            select(SysFile)
            .where(SysFile.md5 == md5, SysFile.deleted == 0)
            .with_for_update()
        )
        return result.scalar_one()

    async def get_by_object_name(
        self,
        db: AsyncSession,
        object_name: str,
    ) -> SysFile | None:
        """根据对象名查询文件"""
        stmt = select(SysFile).where(SysFile.object_name == object_name)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_page(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        keywords: Optional[str] = None,
    ) -> tuple[list[SysFile], int]:
        """
        分页查询文件列表

        Args:
            page: 页码（从 1 开始）
            size: 每页数量
            keywords: 搜索关键词（模糊匹配文件名）

        Returns:
            (items, total) 元组
        """
        stmt = select(SysFile).order_by(SysFile.create_time.desc())

        # 关键词模糊搜索（匹配文件名）
        stmt = self.apply_keyword_filter(
            stmt,
            [SysFile.name],
            keywords,
        )

        return await self.paginate(db, stmt, page, size)

    async def get_all_object_names(
        self,
        db: AsyncSession,
    ) -> list[str]:
        """获取所有文件的 object_name 列表（用于孤儿文件清理比对）"""
        stmt = select(SysFile.object_name)
        result = await db.execute(stmt)
        return [row[0] for row in result.all()]


# 单例
file_repository = FileRepository()
