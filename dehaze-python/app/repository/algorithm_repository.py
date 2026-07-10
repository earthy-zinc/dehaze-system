"""
算法数据访问层
"""

from sqlalchemy import delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_algorithm import SysAlgorithm
from app.repository.base import BaseRepository
from app.repository.base import escape_like


class AlgorithmRepository(BaseRepository[SysAlgorithm]):
    """算法数据访问层"""

    model = SysAlgorithm

    async def get_list_with_keywords(
        self,
        db: AsyncSession,
        keywords: str | None = None,
    ) -> list[SysAlgorithm]:
        """获取算法列表（支持关键词搜索）"""
        stmt = select(SysAlgorithm)
        if keywords:
            stmt = stmt.where(SysAlgorithm.name.like(f"%{escape_like(keywords)}%", escape="\\"))
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_with_children_ids(
        self,
        db: AsyncSession,
        algorithm_ids: list[int],
    ) -> list[int]:
        """获取算法及其子算法 ID（用于删除）"""
        stmt = select(SysAlgorithm.id).where(
            or_(
                SysAlgorithm.id.in_(algorithm_ids),
                SysAlgorithm.parent_id.in_(algorithm_ids),
            )
        )
        result = await db.execute(stmt)
        return [row[0] for row in result.fetchall()]

    async def delete_by_ids(
        self,
        db: AsyncSession,
        ids: list[int],
    ) -> int:
        """根据 ID 列表批量删除"""
        if not ids:
            return 0
        stmt = delete(SysAlgorithm).where(SysAlgorithm.id.in_(ids))
        result = await db.execute(stmt)
        return result.rowcount

    async def get_algorithm_options(
        self,
        db: AsyncSession,
    ) -> list[dict]:
        """获取算法下拉选项列表（树形结构）"""
        stmt = select(SysAlgorithm).where(SysAlgorithm.status == 1)
        result = await db.execute(stmt)
        algorithms = result.scalars().all()

        algorithm_dict = {
            algorithm.id: {"value": algorithm.id, "label": algorithm.name, "children": []}
            for algorithm in algorithms
        }

        root_options = []
        for algorithm in algorithms:
            parent_id = getattr(algorithm, "parent_id")
            if parent_id == 0:
                root_options.append(algorithm_dict[algorithm.id])
            else:
                parent = algorithm_dict.get(parent_id)
                if parent:
                    parent["children"].append(algorithm_dict[algorithm.id])

        return root_options


# 单例
algorithm_repository = AlgorithmRepository()
