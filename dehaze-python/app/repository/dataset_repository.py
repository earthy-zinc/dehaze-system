"""
数据集数据访问层

最复杂的 Repository，包含数据集、数据项、文件关联的查询
"""

from typing import Any

from app.models.entity.sys_dataset import (SysDataset, SysDatasetItem,
                                           SysItemFile)
from app.models.entity.sys_file import SysFile
from app.repository.base import BaseRepository, escape_like
from app.utils.tree import bfs_collect_ids
from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession


class DatasetRepository(BaseRepository[SysDataset]):
    """数据集数据访问层"""

    model = SysDataset

    async def _build_children_map(
        self,
        db: AsyncSession,
    ) -> tuple[dict[int, list[SysDataset]], list[SysDataset]]:
        """构建父子关系映射（内部方法）"""
        stmt = select(SysDataset).where(SysDataset.deleted == 0)
        result = await db.execute(stmt)
        all_datasets = list(result.scalars().all())

        children_map: dict[int, list[SysDataset]] = {}
        for dataset in all_datasets:
            parent_id = int(dataset.parent_id)
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(dataset)

        return children_map, all_datasets

    async def get_leaf_ids(
        self,
        db: AsyncSession,
        dataset_id: int,
    ) -> list[int]:
        """获取指定节点下的所有叶子节点 ID"""
        children_map, _ = await self._build_children_map(db)
        all_ids = bfs_collect_ids(dataset_id, children_map, include_start=True)

        # 找出叶子节点（没有子节点的节点）
        leaf_ids = [ds_id for ds_id in all_ids if not children_map.get(ds_id)]
        return leaf_ids if leaf_ids else [dataset_id]

    async def get_all_descendant_ids(
        self,
        db: AsyncSession,
        dataset_id: int,
    ) -> list[int]:
        """获取所有后代节点 ID（不包括自己）"""
        children_map, _ = await self._build_children_map(db)
        return bfs_collect_ids(dataset_id, children_map, include_start=False)

    async def calculate_statistics(
        self,
        db: AsyncSession,
        dataset_ids: list[int],
    ) -> dict[str, Any]:
        """计算指定数据集的统计信息（叶子节点）"""
        if not dataset_ids:
            return self._empty_stats()

        # 并行执行各项统计查询
        item_count = await self._count_items(db, dataset_ids)
        file_count, total_size = await self._count_files_and_size(db, dataset_ids)
        clear_count = await self._count_by_type(db, dataset_ids, "clear")
        hazy_count = await self._count_by_type(db, dataset_ids, "hazy")
        scene_dist = await self._get_distribution(db, dataset_ids, "scene_type", "未分类")
        haze_dist = await self._get_distribution(db, dataset_ids, "haze_level", "未标注")
        format_dist = await self._get_format_distribution(db, dataset_ids)

        return {
            "itemCount": item_count,
            "fileCount": file_count,
            "totalSize": total_size,
            "clearCount": clear_count,
            "hazyCount": hazy_count,
            "sceneDistribution": scene_dist,
            "hazeDistribution": haze_dist,
            "formatDistribution": format_dist,
        }

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        """返回空统计结构"""
        return {
            "itemCount": 0,
            "fileCount": 0,
            "totalSize": 0,
            "clearCount": 0,
            "hazyCount": 0,
            "sceneDistribution": {},
            "hazeDistribution": {},
            "formatDistribution": {},
        }

    async def _count_items(
        self,
        db: AsyncSession,
        dataset_ids: list[int],
    ) -> int:
        """统计数据项总数"""
        stmt = select(func.count(SysDatasetItem.id)).where(
            SysDatasetItem.dataset_id.in_(dataset_ids)
        )
        result = await db.execute(stmt)
        return result.scalar() or 0

    async def _count_files_and_size(
        self,
        db: AsyncSession,
        dataset_ids: list[int],
    ) -> tuple[int, int]:
        """统计文件总数和总大小"""
        stmt = (
            select(func.count(SysItemFile.id), func.sum(SysFile.size))
            .select_from(SysItemFile)
            .join(SysFile, SysItemFile.file_id == SysFile.id)
            .where(
                SysItemFile.item_id.in_(
                    select(SysDatasetItem.id).where(
                        SysDatasetItem.dataset_id.in_(dataset_ids)
                    )
                )
            )
        )
        result = await db.execute(stmt)
        row = result.first()
        if row:
            return row[0] or 0, int(row[1] or 0)
        return 0, 0

    async def _count_by_type(
        self,
        db: AsyncSession,
        dataset_ids: list[int],
        image_type: str,
    ) -> int:
        """按图片类型统计数量（clear/hazy）"""
        if image_type == "clear":
            type_conditions = or_(
                func.lower(SysItemFile.type).like("%clear%"),
                func.lower(SysItemFile.type).like("%clean%"),
                SysItemFile.type.like("%清晰%"),
                SysItemFile.type.like("%无雾%"),
            )
        else:  # hazy
            type_conditions = or_(
                func.lower(SysItemFile.type).like("%haze%"),
                func.lower(SysItemFile.type).like("%hazy%"),
                SysItemFile.type.like("%有雾%"),
            )

        stmt = (
            select(func.count(SysItemFile.id))
            .select_from(SysItemFile)
            .join(SysDatasetItem, SysItemFile.item_id == SysDatasetItem.id)
            .where(
                and_(
                    SysDatasetItem.dataset_id.in_(dataset_ids),
                    type_conditions,
                )
            )
        )
        result = await db.execute(stmt)
        return result.scalar() or 0

    async def _get_distribution(
        self,
        db: AsyncSession,
        dataset_ids: list[int],
        field: str,
        default_label: str,
    ) -> dict[str, int]:
        """获取字段值分布统计"""
        column = getattr(SysItemFile, field)
        stmt = (
            select(
                func.coalesce(column, default_label).label("label"),
                func.count(SysItemFile.id).label("count"),
            )
            .select_from(SysItemFile)
            .join(SysDatasetItem, SysItemFile.item_id == SysDatasetItem.id)
            .where(SysDatasetItem.dataset_id.in_(dataset_ids))
            .group_by(func.coalesce(column, default_label))
        )
        result = await db.execute(stmt)
        dist: dict[str, int] = {}
        for row in result:
            label_val = getattr(row, "label", "")
            count_val = getattr(row, "count", 0)
            dist[str(label_val)] = int(count_val)
        return dist

    async def _get_format_distribution(
        self,
        db: AsyncSession,
        dataset_ids: list[int],
    ) -> dict[str, int]:
        """获取文件格式分布统计"""
        stmt = (
            select(SysFile.type.label("file_type"),
                   func.count(SysFile.id).label("count"))
            .select_from(SysFile)
            .join(SysItemFile, SysFile.id == SysItemFile.file_id)
            .join(SysDatasetItem, SysItemFile.item_id == SysDatasetItem.id)
            .where(SysDatasetItem.dataset_id.in_(dataset_ids))
            .group_by(SysFile.type)
        )
        result = await db.execute(stmt)
        dist: dict[str, int] = {}
        for row in result:
            file_type_val = getattr(row, "file_type", "")
            count_val = getattr(row, "count", 0)
            dist[str(file_type_val)] = int(count_val)
        return dist

    async def get_item_files(
        self,
        db: AsyncSession,
        item_id: int,
    ) -> list[tuple[SysItemFile, SysFile]]:
        """获取数据项关联的文件列表"""
        stmt = (
            select(SysItemFile, SysFile)
            .select_from(SysItemFile)
            .join(SysFile, SysItemFile.file_id == SysFile.id)
            .where(SysItemFile.item_id == item_id)
        )
        result = await db.execute(stmt)
        return [tuple(row) for row in result.all()]

    async def get_items_by_dataset_id(
        self,
        db: AsyncSession,
        dataset_id: int,
    ) -> list[SysDatasetItem]:
        """获取数据集下的所有数据项"""
        stmt = select(SysDatasetItem).where(
            SysDatasetItem.dataset_id == dataset_id)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_item_by_id(
        self,
        db: AsyncSession,
        item_id: int,
    ) -> SysDatasetItem | None:
        """根据ID获取数据项"""
        stmt = select(SysDatasetItem).where(SysDatasetItem.id == item_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_item_files_by_item_id(
        self,
        db: AsyncSession,
        item_id: int,
    ) -> list[SysItemFile]:
        """获取数据项关联的文件记录（仅 SysItemFile）"""
        stmt = select(SysItemFile).where(SysItemFile.item_id == item_id)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_dataset_options(
        self,
        db: AsyncSession,
    ) -> list[dict]:
        """获取数据集下拉选项（树形结构）"""
        children_map, datasets = await self._build_children_map(db)

        # 过滤启用状态
        enabled_datasets = [d for d in datasets if bool(d.status == 1)]
        if not enabled_datasets:
            return []

        # 重新构建启用状态的 children_map
        enabled_children_map: dict[int, list[SysDataset]] = {}
        for dataset in enabled_datasets:
            parent_id = int(dataset.parent_id)
            if parent_id not in enabled_children_map:
                enabled_children_map[parent_id] = []
            enabled_children_map[parent_id].append(dataset)

        def build_options_tree(parent_id: int) -> list[dict]:
            options = []
            for child in enabled_children_map.get(parent_id, []):
                child_id = int(child.id)
                option: dict[str, Any] = {
                    "value": child_id,
                    "label": child.name,
                    "children": build_options_tree(child_id),
                }
                if not option["children"]:
                    del option["children"]
                options.append(option)
            return options

        return build_options_tree(0)

    async def get_list_with_keywords(
        self,
        db: AsyncSession,
        keywords: str | None = None,
    ) -> list[SysDataset]:
        """获取数据集列表（支持关键字搜索）"""
        stmt = select(SysDataset).where(SysDataset.deleted == 0)
        if keywords:
            stmt = stmt.where(SysDataset.name.like(
                f"%{escape_like(keywords)}%", escape="\\"))
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_by_id(
        self,
        db: AsyncSession,
        id: int,
        *,
        with_deleted: bool = False,
    ) -> SysDataset | None:
        """根据ID获取数据集"""
        if with_deleted:
            stmt = select(SysDataset).where(SysDataset.id == id)
        else:
            stmt = select(SysDataset).where(
                and_(SysDataset.id == id, SysDataset.deleted == 0)
            )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_dataset_id(
        self,
        db: AsyncSession,
        dataset_id: int,
        *,
        include_deleted: bool = False,
    ) -> SysDataset | None:
        """根据数据集ID获取数据集（别名方法，保持向后兼容）"""
        return await self.get_by_id(db, dataset_id, with_deleted=include_deleted)

    async def get_children_count(
        self,
        db: AsyncSession,
        dataset_id: int,
    ) -> int:
        """获取子数据集数量"""
        stmt = select(func.count(SysDataset.id)).where(
            SysDataset.parent_id == dataset_id)
        result = await db.execute(stmt)
        return result.scalar() or 0

    async def delete_by_ids(
        self,
        db: AsyncSession,
        ids: list[int],
    ) -> int:
        """批量删除数据集"""
        if not ids:
            return 0
        stmt = delete(SysDataset).where(SysDataset.id.in_(ids))
        result = await db.execute(stmt)
        return result.rowcount

    async def check_name_exists(
        self,
        db: AsyncSession,
        parent_id: int,
        name: str,
        exclude_id: int | None = None,
    ) -> bool:
        """检查同一父节点下名称是否已存在"""
        stmt = select(func.count(SysDataset.id)).where(
            and_(
                SysDataset.parent_id == parent_id,
                SysDataset.name == name,
                SysDataset.deleted == 0,
            )
        )
        if exclude_id is not None:
            stmt = stmt.where(SysDataset.id != exclude_id)
        result = await db.execute(stmt)
        return (result.scalar() or 0) > 0

    async def get_items_count(
        self,
        db: AsyncSession,
        dataset_ids: list[int],
        keywords: str | None = None,
    ) -> int:
        """获取数据项总数（支持关键词过滤）"""
        if not dataset_ids:
            return 0
        stmt = select(func.count(SysDatasetItem.id)).where(
            SysDatasetItem.dataset_id.in_(dataset_ids)
        )
        if keywords:
            stmt = stmt.where(
                SysDatasetItem.name.like(
                    f"%{escape_like(keywords)}%", escape="\\")
            )
        result = await db.execute(stmt)
        return result.scalar() or 0

    async def get_items_paginated(
        self,
        db: AsyncSession,
        dataset_ids: list[int],
        offset: int,
        limit: int,
        keywords: str | None = None,
    ) -> list[SysDatasetItem]:
        """分页获取数据项（支持关键词过滤）"""
        stmt = select(SysDatasetItem).where(
            SysDatasetItem.dataset_id.in_(dataset_ids)
        )
        if keywords:
            stmt = stmt.where(
                SysDatasetItem.name.like(
                    f"%{escape_like(keywords)}%", escape="\\")
            )
        stmt = stmt.order_by(SysDatasetItem.id.desc()
                             ).offset(offset).limit(limit)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_item_ids_by_dataset(
        self,
        db: AsyncSession,
        dataset_id: int,
    ) -> list[int]:
        """获取数据集下所有数据项ID"""
        stmt = select(SysDatasetItem.id).where(
            SysDatasetItem.dataset_id == dataset_id)
        result = await db.execute(stmt)
        return [row[0] for row in result.all()]

    async def create_item(
        self,
        db: AsyncSession,
        dataset_id: int,
        name: str,
    ) -> SysDatasetItem:
        """创建数据项"""
        item = SysDatasetItem(dataset_id=dataset_id, name=name)
        db.add(item)
        await db.flush()
        return item

    async def update_item(
        self,
        db: AsyncSession,
        item_id: int,
        name: str,
    ) -> bool:
        """更新数据项名称"""
        stmt = select(SysDatasetItem).where(SysDatasetItem.id == item_id)
        result = await db.execute(stmt)
        item = result.scalar_one_or_none()
        if not item:
            return False
        item.name = name
        return True

    async def delete_item_by_id(
        self,
        db: AsyncSession,
        item_id: int,
    ) -> bool:
        """删除单个数据项"""
        stmt = delete(SysDatasetItem).where(SysDatasetItem.id == item_id)
        result = await db.execute(stmt)
        return result.rowcount > 0

    async def delete_items_by_dataset_id(
        self,
        db: AsyncSession,
        dataset_id: int,
    ) -> int:
        """删除数据集下所有数据项"""
        stmt = delete(SysDatasetItem).where(
            SysDatasetItem.dataset_id == dataset_id)
        result = await db.execute(stmt)
        return result.rowcount

    async def delete_item_files_by_item_id(
        self,
        db: AsyncSession,
        item_id: int,
    ) -> int:
        """删除数据项关联的文件记录"""
        stmt = delete(SysItemFile).where(SysItemFile.item_id == item_id)
        result = await db.execute(stmt)
        return result.rowcount

    async def delete_item_files_by_item_ids(
        self,
        db: AsyncSession,
        item_ids: list[int],
    ) -> int:
        """批量删除数据项关联的文件记录"""
        if not item_ids:
            return 0
        stmt = delete(SysItemFile).where(SysItemFile.item_id.in_(item_ids))
        result = await db.execute(stmt)
        return result.rowcount

    async def get_item_with_files(
        self,
        db: AsyncSession,
        item_id: int,
    ) -> tuple[SysDatasetItem | None, list[tuple[SysItemFile, SysFile]]]:
        """获取数据项及其关联文件"""
        # 获取数据项
        stmt = select(SysDatasetItem).where(SysDatasetItem.id == item_id)
        result = await db.execute(stmt)
        item = result.scalar_one_or_none()

        if not item:
            return None, []

        # 获取关联文件
        stmt = (
            select(SysItemFile, SysFile)
            .select_from(SysItemFile)
            .join(SysFile, SysItemFile.file_id == SysFile.id)
            .where(SysItemFile.item_id == item_id)
        )
        result = await db.execute(stmt)
        item_files = [tuple(row) for row in result.all()]

        return item, item_files

    async def get_dataset_tree_path(
        self,
        db: AsyncSession,
        dataset_id: int,
    ) -> str | None:
        """获取数据集的树路径"""
        stmt = select(SysDataset.tree_path).where(SysDataset.id == dataset_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def update_tree_path(
        self,
        db: AsyncSession,
        dataset_id: int,
        tree_path: str,
    ) -> bool:
        """更新数据集树路径"""
        stmt = select(SysDataset).where(SysDataset.id == dataset_id)
        result = await db.execute(stmt)
        dataset = result.scalar_one_or_none()
        if not dataset:
            return False
        dataset.tree_path = tree_path
        return True

    async def get_all_datasets_for_tree_path_update(
        self,
        db: AsyncSession,
        dataset_ids: list[int],
    ) -> list[SysDataset]:
        """获取需要更新树路径的数据集列表"""
        if not dataset_ids:
            return []
        stmt = select(SysDataset).where(SysDataset.id.in_(dataset_ids))
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_dataset_depth(
        self,
        db: AsyncSession,
        dataset_ids: list[int],
    ) -> dict[int, int]:
        """获取数据集的深度（用于排序删除）"""
        if not dataset_ids:
            return {}
        stmt = select(SysDataset.id, SysDataset.tree_path).where(
            SysDataset.id.in_(dataset_ids))
        result = await db.execute(stmt)
        depth_map = {}
        for row in result:
            tree_path = row.tree_path or ""
            depth_map[row.id] = tree_path.count(",") if tree_path else 0
        return depth_map

    # ── ItemFile 相关方法 ─────────────────────────────

    async def get_item_file_by_id(
        self,
        db: AsyncSession,
        file_id: int,
    ) -> SysItemFile | None:
        """根据ID获取图片文件关联记录"""
        stmt = select(SysItemFile).where(SysItemFile.id == file_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_item_file_with_file(
        self,
        db: AsyncSession,
        file_id: int,
    ) -> tuple[SysItemFile, SysFile] | None:
        """获取图片文件关联记录及对应的文件信息"""
        stmt = (
            select(SysItemFile, SysFile)
            .select_from(SysItemFile)
            .join(SysFile, SysItemFile.file_id == SysFile.id)
            .where(SysItemFile.id == file_id)
        )
        result = await db.execute(stmt)
        row = result.first()
        if not row:
            return None
        return tuple(row)

    async def delete_item_file_by_id(
        self,
        db: AsyncSession,
        file_id: int,
    ) -> bool:
        """删除单个图片文件关联记录"""
        stmt = delete(SysItemFile).where(SysItemFile.id == file_id)
        result = await db.execute(stmt)
        return result.rowcount > 0

    async def delete_item_files_by_ids(
        self,
        db: AsyncSession,
        file_ids: list[int],
    ) -> int:
        """批量删除图片文件关联记录"""
        if not file_ids:
            return 0
        stmt = delete(SysItemFile).where(SysItemFile.id.in_(file_ids))
        result = await db.execute(stmt)
        return result.rowcount


# 单例
dataset_repository = DatasetRepository()
