"""
部门数据访问层
"""

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_dept import SysDept
from app.repository.base import BaseRepository, escape_like
from app.utils.tree import generate_tree_path as gen_tree_path


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
            stmt = stmt.where(SysDept.name.like(f"%{escape_like(keywords)}%", escape="\\"))
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

    async def get_dept_options_tree(
        self,
        db: AsyncSession,
    ) -> list[dict[str, Any]]:
        """获取部门下拉选项列表（树形结构）"""
        stmt = (
            select(SysDept).where(SysDept.deleted == 0, SysDept.status == 1).order_by(SysDept.sort)
        )
        result = await db.execute(stmt)
        dept_list = result.scalars().all()

        if not dept_list:
            return []

        dept_dict: dict[int, dict[str, Any]] = {
            dept.id: {"value": dept.id, "label": dept.name, "children": []} for dept in dept_list
        }

        root_options: list[dict[str, Any]] = []
        for dept in dept_list:
            parent_id = dept.parent_id
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
        parent_tree_path = parent_dept.tree_path if parent_dept else None
        return gen_tree_path(parent_tree_path, parent_id)

    async def count_children_by_parents(
        self,
        db: AsyncSession,
        parent_ids: list[int],
    ) -> dict[int, int]:
        """统计每个父部门下的直接子部门数量（deleted=0，避免 N+1）

        Returns:
            {parent_id: 子部门数量}
        """
        if not parent_ids:
            return {}
        stmt = (
            select(SysDept.parent_id, func.count())
            .where(SysDept.parent_id.in_(parent_ids), SysDept.deleted == 0)
            .group_by(SysDept.parent_id)
        )
        result = await db.execute(stmt)
        return {int(row[0]): int(row[1]) for row in result.fetchall()}


# 单例
dept_repository = DeptRepository()
