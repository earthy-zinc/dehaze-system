"""
部门数据访问层
"""

from typing import Any

from app.models.entity.sys_dept import SysDept
from app.repository.base import BaseRepository, escape_like
from app.utils.tree import generate_tree_path as gen_tree_path
from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.expression import BinaryExpression


class DeptRepository(BaseRepository[SysDept]):
    """部门数据访问层"""

    model = SysDept

    async def get_children_ids(
        self,
        db: AsyncSession,
        dept_id: int,
    ) -> list[int]:
        """查询部门及所有子部门 ID（基于 tree_path LIKE）"""
        dept = await self.get_by_id(db, dept_id)
        if not dept:
            return [dept_id]
        stmt = select(SysDept.id).where(
            SysDept.tree_path.like(f"{dept.tree_path}%"),
            SysDept.deleted == 0,
        )
        result = await db.execute(stmt)
        child_ids = [int(row[0]) for row in result.fetchall()]
        return [int(dept.id)] + child_ids

    async def count_children(
        self,
        db: AsyncSession,
        dept_id: int,
    ) -> int:
        """统计直接子部门数量"""
        stmt = select(func.count()).select_from(SysDept).where(
            SysDept.parent_id == dept_id,
            SysDept.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar() or 0

    async def get_max_child_depth(
        self,
        db: AsyncSession,
        dept_id: int,
    ) -> int:
        """获取子部门最大深度"""
        dept = await self.get_by_id(db, dept_id)
        if not dept:
            return 0

        # 查询所有子部门的 tree_path
        stmt = select(SysDept.tree_path).where(
            SysDept.tree_path.like(f"{dept.tree_path}%"),
            SysDept.id != dept_id,
            SysDept.deleted == 0,
        )
        result = await db.execute(stmt)
        tree_paths = [row[0] for row in result.fetchall() if row[0]]

        if not tree_paths:
            return 0

        # 计算最大深度
        max_depth = 0
        for path in tree_paths:
            depth = len(path.split(",")) if path else 0
            max_depth = max(max_depth, depth)

        return max_depth

    async def check_name_exists(
        self,
        db: AsyncSession,
        name: str,
        *,
        exclude_id: int | None = None,
        parent_id: int | None = None,
    ) -> bool:
        """检查部门名称是否重复（同层级内）"""
        stmt = select(SysDept).where(
            SysDept.name == name,
            SysDept.deleted == 0,
        )
        if exclude_id:
            stmt = stmt.where(SysDept.id != exclude_id)
        if parent_id is not None:
            stmt = stmt.where(SysDept.parent_id == parent_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def get_dept_list(
        self,
        db: AsyncSession,
        keywords: str | None = None,
        status: int | None = None,
    ) -> list[SysDept]:
        """获取部门列表"""
        stmt = select(SysDept).where(SysDept.deleted == 0)

        if keywords:
            stmt = stmt.where(SysDept.name.like(
                f"%{escape_like(keywords)}%", escape="\\"))
        if status is not None:
            stmt = stmt.where(SysDept.status == status)

        stmt = stmt.order_by(SysDept.sort, SysDept.create_time.desc())
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_dept_form(
        self,
        db: AsyncSession,
        dept_id: int,
    ) -> dict[str, Any] | None:
        """获取部门表单数据"""
        dept = await self.get_by_id(db, dept_id)
        if not dept:
            return None

        return {
            "id": dept.id,
            "name": dept.name,
            "parentId": dept.parent_id,
            "status": dept.status,
            "sort": dept.sort,
        }

    async def get_dept_options(
        self,
        db: AsyncSession,
    ) -> list[dict[str, Any]]:
        """获取部门下拉选项列表（平铺格式）"""
        stmt = (
            select(SysDept)
            .where(SysDept.deleted == 0, SysDept.status == 1)
            .order_by(SysDept.sort)
        )
        result = await db.execute(stmt)
        depts = result.scalars().all()
        return [{"value": dept.id, "label": dept.name} for dept in depts]

    async def get_dept_options_tree(
        self,
        db: AsyncSession,
    ) -> list[dict[str, Any]]:
        """获取部门下拉选项列表（树形结构）"""
        stmt = (
            select(SysDept)
            .where(SysDept.deleted == 0, SysDept.status == 1)
            .order_by(SysDept.sort)
        )
        result = await db.execute(stmt)
        dept_list = result.scalars().all()

        if not dept_list:
            return []

        dept_dict: dict[int, dict[str, Any]] = {dept.id: {"value": dept.id,
                                                          "label": dept.name, "children": []} for dept in dept_list}

        root_options: list[dict[str, Any]] = []
        for dept in dept_list:
            parent_id = getattr(dept, "parent_id", 0)
            if parent_id == 0:
                root_options.append(dept_dict[dept.id])
            else:
                parent = dept_dict.get(parent_id)
                if parent:
                    parent["children"].append(dept_dict[dept.id])

        return root_options

    async def generate_tree_path(
        self,
        db: AsyncSession,
        parent_id: int,
    ) -> str:
        """生成部门路径"""
        if parent_id == 0:
            return "0"
        parent_dept = await self.get_by_id(db, parent_id)
        parent_tree_path = getattr(
            parent_dept, "tree_path", None) if parent_dept else None
        return gen_tree_path(parent_tree_path, parent_id)

    async def delete_depts(
        self,
        db: AsyncSession,
        dept_ids: list[int],
    ) -> int:
        """删除部门（物理删除）"""
        if not dept_ids:
            return 0

        stmt = delete(SysDept).where(SysDept.id.in_(dept_ids))
        result = await db.execute(stmt)
        return result.rowcount

    async def delete_dept_with_children(
        self,
        db: AsyncSession,
        dept_id: int,
    ) -> int:
        """删除部门及其所有子部门（基于 tree_path LIKE 级联删除，匹配 Java 行为）"""
        stmt = delete(SysDept).where(
            or_(
                SysDept.id == dept_id,
                SysDept.tree_path.like(f"%,{dept_id},%"),
                SysDept.tree_path.like(f"{dept_id},%"),
                SysDept.tree_path.like(f"%,{dept_id}"),
            )
        )
        result = await db.execute(stmt)
        return result.rowcount

    async def delete_depts_with_children(
        self,
        db: AsyncSession,
        dept_ids: list[int],
    ) -> int:
        """删除部门（包含子部门）"""
        if not dept_ids:
            return 0

        # 构建批量删除条件：删除指定部门及其所有子部门
        conditions: list[BinaryExpression] = [SysDept.id.in_(dept_ids)]
        for dept_id in dept_ids:
            conditions.append(SysDept.tree_path.like(f"%,{dept_id},%"))
            conditions.append(SysDept.tree_path.like(f"{dept_id},%"))
            conditions.append(SysDept.tree_path.like(f"%,{dept_id}"))

        stmt = delete(SysDept).where(or_(*conditions))
        result = await db.execute(stmt)
        return result.rowcount


# 单例
dept_repository = DeptRepository()
