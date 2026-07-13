"""
Repository 泛型基类

提供通用的 CRUD 操作，子类只需声明 model 类型即可继承全部能力。
特定查询在子类中扩展。
"""

from typing import TYPE_CHECKING, Any, Generic, TypeVar

from sqlalchemy import Integer, Select, delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql.expression import ColumnElement
from sqlalchemy.sql.sqltypes import Integer as SQLInteger

if TYPE_CHECKING:
    from sqlalchemy import Column as ColumnType

T = TypeVar("T", bound=DeclarativeBase)


def escape_like(pattern: str) -> str:
    """
    转义 SQL LIKE 查询中的特殊字符

    Usage:
        stmt = stmt.where(User.name.like(f"%{escape_like(keywords)}%", escape="\\"))
    """
    if not pattern:
        return ""
    return pattern.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")


class BaseRepository(Generic[T]):
    """
    泛型 Repository 基类

    使用方式:
        class UserRepository(BaseRepository[SysUser]):
            model = SysUser
    """

    model: type[T]

    # ── 查询 ──────────────────────────────────────────

    def _get_id_column(self) -> ColumnElement[Integer]:
        """获取 ID 列"""
        return getattr(self.model, "id")

    def _get_deleted_column(self) -> ColumnElement[Integer]:
        """获取 deleted 列"""
        return getattr(self.model, "deleted")

    async def get_by_id(
        self,
        db: AsyncSession,
        id: int,
        *,
        with_deleted: bool = False,
    ) -> T | None:
        """根据 ID 查询单条记录"""
        id_column = self._get_id_column()
        stmt = select(self.model).where(id_column == id)
        if not with_deleted and hasattr(self.model, "deleted"):
            deleted_column = self._get_deleted_column()
            stmt = stmt.where(deleted_column == 0)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_ids(
        self,
        db: AsyncSession,
        ids: list[int],
        *,
        with_deleted: bool = False,
    ) -> list[T]:
        """根据 ID 列表批量查询记录"""
        if not ids:
            return []
        id_column = self._get_id_column()
        stmt = select(self.model).where(id_column.in_(ids))
        if not with_deleted and hasattr(self.model, "deleted"):
            deleted_column = self._get_deleted_column()
            stmt = stmt.where(deleted_column == 0)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_all(
        self,
        db: AsyncSession,
        *,
        with_deleted: bool = False,
    ) -> list[T]:
        """查询全部记录"""
        stmt = select(self.model)
        if not with_deleted and hasattr(self.model, "deleted"):
            deleted_column = self._get_deleted_column()
            stmt = stmt.where(deleted_column == 0)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def paginate(
        self,
        db: AsyncSession,
        stmt: Select,
        page: int,
        size: int,
    ) -> tuple[list[T], int]:
        """
        对已构建的查询执行分页

        Args:
            stmt: 已构建好条件的 select 语句
            page: 页码（从 1 开始）
            size: 每页数量

        Returns:
            (items, total) 元组
        """
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        paged_stmt = stmt.offset((page - 1) * size).limit(size)
        result = await db.execute(paged_stmt)
        items = list(result.scalars().all())
        return items, total

    @staticmethod
    async def paginate_rows(
        db: AsyncSession,
        stmt: Select,
        page: int,
        size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        对列级查询（select(columns)）执行分页，返回 dict 列表

        Args:
            stmt: 已构建好条件的 select 语句
            page: 页码（从 1 开始）
            size: 每页数量

        Returns:
            (rows, total) 元组，rows 为 dict 列表
        """
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        paged_stmt = stmt.offset((page - 1) * size).limit(size)
        result = await db.execute(paged_stmt)
        columns = list(result.keys())
        rows = [dict(zip(columns, row)) for row in result.all()]
        return rows, total

    @staticmethod
    def apply_keyword_filter(
        stmt: Select,
        columns: list,
        keyword: str | None,
    ) -> Select:
        """为查询添加多列 LIKE 模糊搜索"""
        if not keyword:
            return stmt
        escaped = escape_like(keyword)
        conditions = [col.like(f"%{escaped}%", escape="\\") for col in columns]
        return stmt.where(or_(*conditions))

    # ── 写入 ──────────────────────────────────────────

    async def create(self, db: AsyncSession, entity: T) -> T:
        """插入实体"""
        db.add(entity)
        await db.flush()
        await db.refresh(entity)
        return entity

    async def create_all(self, db: AsyncSession, entities: list[T]) -> list[T]:
        """批量插入"""
        db.add_all(entities)
        await db.flush()
        return entities

    async def save(self, db: AsyncSession, entity: T) -> T:
        """保存（merge: 有 ID 则更新，无则插入）"""
        merged = await db.merge(entity)
        await db.flush()
        return merged

    async def update(
        self,
        db: AsyncSession,
        entity: T,
        data: dict[str, Any],
    ) -> T:
        """更新实体字段"""
        for key, value in data.items():
            if hasattr(entity, key):
                setattr(entity, key, value)
        await db.flush()
        await db.refresh(entity)
        return entity

    # ── 删除 ──────────────────────────────────────────

    async def delete_by_ids(
        self,
        db: AsyncSession,
        ids: list[int],
    ) -> int:
        """按 ID 列表批量硬删除"""
        if not ids:
            return 0
        id_column = self._get_id_column()
        stmt = delete(self.model).where(id_column.in_(ids))
        result = await db.execute(stmt)
        return result.rowcount

    async def soft_delete_by_ids(
        self,
        db: AsyncSession,
        ids: list[int],
    ) -> int:
        """按 ID 列表批量软删除（将 deleted 置为 1）"""
        if not ids:
            return 0
        if not hasattr(self.model, "deleted"):
            raise AttributeError(
                f"{self.model.__name__} does not have 'deleted' field")
        id_column = self._get_id_column()
        stmt = (
            update(self.model)
            .where(id_column.in_(ids))
            .values(deleted=1)
        )
        result = await db.execute(stmt)
        return result.rowcount

    # ── 计数 ──────────────────────────────────────────

    async def count(self, db: AsyncSession, stmt: Select | None = None) -> int:
        """统计数量"""
        if stmt is None:
            base_stmt = select(self.model)
            if hasattr(self.model, "deleted"):
                deleted_column = self._get_deleted_column()
                base_stmt = base_stmt.where(deleted_column == 0)
            count_stmt = select(func.count()).select_from(base_stmt.subquery())
        else:
            count_stmt = select(func.count()).select_from(stmt.subquery())
        return (await db.execute(count_stmt)).scalar() or 0

    async def exists_by_id(self, db: AsyncSession, id: int) -> bool:
        """判断 ID 是否存在"""
        return await self.get_by_id(db, id) is not None
