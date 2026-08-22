"""
数据集数据访问层

最复杂的 Repository，包含数据集、数据项、文件关联的查询
"""

from typing import Any

from sqlalchemy import and_, case, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_dataset import SysDataset, SysDatasetItem, SysItemFile
from app.models.entity.sys_file import SysFile
from app.repository.base import BaseRepository
from app.utils.tree import bfs_collect_ids


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

    async def get_all_descendant_ids_batch(
        self,
        db: AsyncSession,
        dataset_ids: list[int],
    ) -> dict[int, list[int]]:
        """批量获取多个数据集的后代 ID（含自身，1 次全表查询替代 N 次，避免 N+1）

        Returns:
            dict: {dataset_id: [自身ID + 所有后代ID]}
        """
        if not dataset_ids:
            return {}
        children_map, _ = await self._build_children_map(db)
        result: dict[int, list[int]] = {}
        for dataset_id in dataset_ids:
            result[dataset_id] = bfs_collect_ids(dataset_id, children_map, include_start=True)
        return result

    async def get_items_by_dataset_id(
        self,
        db: AsyncSession,
        dataset_id: int,
    ) -> list[SysDatasetItem]:
        """获取数据集下的所有数据项"""
        stmt = select(SysDatasetItem).where(SysDatasetItem.dataset_id == dataset_id)
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

    async def get_item_files_by_item_ids(
        self,
        db: AsyncSession,
        item_ids: list[int],
    ) -> dict[int, list[SysItemFile]]:
        """批量获取多个数据项的文件记录（按 item_id 分组，避免 N+1）"""
        if not item_ids:
            return {}
        stmt = select(SysItemFile).where(SysItemFile.item_id.in_(item_ids))
        result = await db.execute(stmt)
        files_map: dict[int, list[SysItemFile]] = {}
        for item_file in result.scalars().all():
            iid = int(item_file.item_id)
            if iid not in files_map:
                files_map[iid] = []
            files_map[iid].append(item_file)
        return files_map

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

    async def find_datasets_with_clear_gt(
        self,
        db: AsyncSession,
        task_type: str | None = None,
    ) -> list[SysDataset]:
        """查询含清晰图 GT（type=clear）的数据集，用于算法评估测试集选项。

        - 仅返回启用（status=1）且未删除的数据集
        - 按 task_type 过滤时匹配数据集 type 字段
        - 数据集自身存在 type=clear 的 item_file 即视为含 GT
        """
        stmt = (
            select(SysDataset)
            .join(SysDatasetItem, SysDatasetItem.dataset_id == SysDataset.id)
            .join(SysItemFile, SysItemFile.item_id == SysDatasetItem.id)
            .where(
                SysDataset.deleted == 0,
                SysDataset.status == 1,
                SysItemFile.type == "clear",
            )
            .distinct()
        )
        if task_type:
            stmt = stmt.where(SysDataset.type == task_type)
        stmt = stmt.order_by(SysDataset.id)
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
            stmt = select(SysDataset).where(and_(SysDataset.id == id, SysDataset.deleted == 0))
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_children_count(
        self,
        db: AsyncSession,
        dataset_id: int,
    ) -> int:
        """获取子数据集数量"""
        stmt = select(func.count(SysDataset.id)).where(SysDataset.parent_id == dataset_id)
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
            stmt = stmt.where(SysDatasetItem.name.like(f"%{keywords}%"))
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
        stmt = select(SysDatasetItem).where(SysDatasetItem.dataset_id.in_(dataset_ids))
        if keywords:
            stmt = stmt.where(SysDatasetItem.name.like(f"%{keywords}%"))
        stmt = stmt.order_by(SysDatasetItem.id.desc()).offset(offset).limit(limit)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_item_ids_by_dataset_ids(
        self,
        db: AsyncSession,
        dataset_ids: list[int],
    ) -> list[int]:
        """批量获取多个数据集下的所有数据项 ID（避免 N+1）"""
        if not dataset_ids:
            return []
        stmt = select(SysDatasetItem.id).where(SysDatasetItem.dataset_id.in_(dataset_ids))
        result = await db.execute(stmt)
        return [row[0] for row in result.all()]

    async def delete_item_by_id(
        self,
        db: AsyncSession,
        item_id: int,
    ) -> bool:
        """删除单个数据项"""
        stmt = delete(SysDatasetItem).where(SysDatasetItem.id == item_id)
        result = await db.execute(stmt)
        return result.rowcount > 0

    async def get_items_by_ids(
        self,
        db: AsyncSession,
        item_ids: list[int],
    ) -> list[SysDatasetItem]:
        """批量根据 ID 获取数据项"""
        if not item_ids:
            return []
        stmt = select(SysDatasetItem).where(SysDatasetItem.id.in_(item_ids))
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def delete_items_by_ids(
        self,
        db: AsyncSession,
        item_ids: list[int],
    ) -> int:
        """批量删除数据项"""
        if not item_ids:
            return 0
        stmt = delete(SysDatasetItem).where(SysDatasetItem.id.in_(item_ids))
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

    async def get_datasets_by_ids(
        self,
        db: AsyncSession,
        dataset_ids: list[int],
    ) -> list[SysDataset]:
        """获取指定数据集列表（用于级联删除等场景）"""
        if not dataset_ids:
            return []
        stmt = select(SysDataset).where(SysDataset.id.in_(dataset_ids))
        result = await db.execute(stmt)
        return list(result.scalars().all())

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

    async def get_item_files_by_ids(
        self,
        db: AsyncSession,
        file_ids: list[int],
    ) -> list[SysItemFile]:
        """批量根据 ID 获取图片文件关联记录"""
        if not file_ids:
            return []
        stmt = select(SysItemFile).where(SysItemFile.id.in_(file_ids))
        result = await db.execute(stmt)
        return list(result.scalars().all())

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

    async def find_root_page(
        self,
        db: AsyncSession,
        page_num: int = 1,
        page_size: int = 10,
        keyword: str | None = None,
        type: str | None = None,
        status: int | None = None,
    ) -> tuple[list[SysDataset], int]:
        """分页查询根节点数据集（parent_id=0），支持 keyword/type/status 过滤"""
        base_stmt = select(SysDataset).where(
            and_(SysDataset.parent_id == 0, SysDataset.deleted == 0)
        )
        count_stmt = select(func.count(SysDataset.id)).where(
            and_(SysDataset.parent_id == 0, SysDataset.deleted == 0)
        )

        if keyword:
            keyword_filter = SysDataset.name.like(f"%{keyword}%")
            base_stmt = base_stmt.where(keyword_filter)
            count_stmt = count_stmt.where(keyword_filter)
        if type:
            base_stmt = base_stmt.where(SysDataset.type == type)
            count_stmt = count_stmt.where(SysDataset.type == type)
        if status is not None:
            base_stmt = base_stmt.where(SysDataset.status == status)
            count_stmt = count_stmt.where(SysDataset.status == status)

        count_result = await db.execute(count_stmt)
        total = count_result.scalar() or 0

        offset = (page_num - 1) * page_size
        stmt = base_stmt.order_by(SysDataset.id.desc()).offset(offset).limit(page_size)
        result = await db.execute(stmt)
        items = list(result.scalars().all())

        return items, total

    async def find_by_parent_id(
        self,
        db: AsyncSession,
        parent_id: int,
    ) -> list[SysDataset]:
        """查询指定父节点的直接子节点"""
        stmt = (
            select(SysDataset)
            .where(and_(SysDataset.parent_id == parent_id, SysDataset.deleted == 0))
            .order_by(SysDataset.id)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def find_by_parent_ids(
        self,
        db: AsyncSession,
        parent_ids: list[int],
    ) -> list[SysDataset]:
        """批量查询多个父节点的直接子节点"""
        if not parent_ids:
            return []
        stmt = (
            select(SysDataset)
            .where(and_(SysDataset.parent_id.in_(parent_ids), SysDataset.deleted == 0))
            .order_by(SysDataset.id)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def count_has_children(
        self,
        db: AsyncSession,
        parent_ids: list[int],
    ) -> dict[int, bool]:
        """批量查询哪些节点有子节点"""
        if not parent_ids:
            return {}
        stmt = (
            select(SysDataset.parent_id, func.count(SysDataset.id).label("cnt"))
            .where(and_(SysDataset.parent_id.in_(parent_ids), SysDataset.deleted == 0))
            .group_by(SysDataset.parent_id)
        )
        result = await db.execute(stmt)
        has_children_map: dict[int, bool] = {}
        for row in result:
            has_children_map[int(row.parent_id)] = row.cnt > 0
        for pid in parent_ids:
            if pid not in has_children_map:
                has_children_map[pid] = False
        return has_children_map

    async def find_all(
        self,
        db: AsyncSession,
    ) -> list[SysDataset]:
        """查询所有未删除的数据集"""
        stmt = select(SysDataset).where(SysDataset.deleted == 0).order_by(SysDataset.id)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def count_items_per_dataset(
        self,
        db: AsyncSession,
        dataset_ids: list[int],
    ) -> dict[int, int]:
        """批量统计每个数据集的数据项数量"""
        if not dataset_ids:
            return {}
        stmt = (
            select(SysDatasetItem.dataset_id, func.count(SysDatasetItem.id).label("cnt"))
            .where(SysDatasetItem.dataset_id.in_(dataset_ids))
            .group_by(SysDatasetItem.dataset_id)
        )
        result = await db.execute(stmt)
        count_map: dict[int, int] = {}
        for row in result:
            count_map[int(row.dataset_id)] = int(row.cnt)
        return count_map

    async def count_dataset_stats_batch(
        self,
        db: AsyncSession,
        dataset_ids: list[int],
    ) -> dict[int, dict[str, int]]:
        """批量统计每个数据集的文件数、总大小、已标注数、未标注数"""
        if not dataset_ids:
            return {}
        annotated_cond = and_(
            SysItemFile.haze_level.isnot(None),
            SysItemFile.haze_level != "",
        )
        unannotated_cond = or_(
            SysItemFile.haze_level.is_(None),
            SysItemFile.haze_level == "",
        )
        stmt = (
            select(
                SysDatasetItem.dataset_id.label("dataset_id"),
                func.count(SysItemFile.id).label("file_count"),
                func.coalesce(func.sum(SysFile.size), 0).label("total_size"),
                func.sum(case((annotated_cond, 1), else_=0)).label("annotated_count"),
                func.sum(case((unannotated_cond, 1), else_=0)).label("unannotated_count"),
            )
            .select_from(SysDatasetItem)
            .outerjoin(SysItemFile, SysItemFile.item_id == SysDatasetItem.id)
            .outerjoin(SysFile, SysFile.id == SysItemFile.file_id)
            .where(SysDatasetItem.dataset_id.in_(dataset_ids))
            .group_by(SysDatasetItem.dataset_id)
        )
        result = await db.execute(stmt)
        stats_map: dict[int, dict[str, int]] = {}
        for row in result:
            ds_id = int(row.dataset_id)
            stats_map[ds_id] = {
                "fileCount": int(row.file_count or 0),
                "totalSize": int(row.total_size or 0),
                "annotatedCount": int(row.annotated_count or 0),
                "unannotatedCount": int(row.unannotated_count or 0),
            }
        return stats_map

    async def count_scene_distribution_batch(
        self,
        db: AsyncSession,
        dataset_ids: list[int],
    ) -> dict[int, dict[str, int]]:
        """批量统计每个数据集的场景类型分布"""
        return await self._count_distribution_batch(
            db, dataset_ids, SysItemFile.scene_type, "未分类"
        )

    async def count_haze_distribution_batch(
        self,
        db: AsyncSession,
        dataset_ids: list[int],
    ) -> dict[int, dict[str, int]]:
        """批量统计每个数据集的雾霾程度分布"""
        return await self._count_distribution_batch(
            db, dataset_ids, SysItemFile.haze_level, "未标注"
        )

    async def count_format_distribution_batch(
        self,
        db: AsyncSession,
        dataset_ids: list[int],
    ) -> dict[int, dict[str, int]]:
        """批量统计每个数据集的文件格式分布"""
        if not dataset_ids:
            return {}
        stmt = (
            select(
                SysDatasetItem.dataset_id.label("dataset_id"),
                func.coalesce(SysFile.type, "unknown").label("key"),
                func.count(SysFile.id).label("cnt"),
            )
            .select_from(SysDatasetItem)
            .join(SysItemFile, SysItemFile.item_id == SysDatasetItem.id)
            .join(SysFile, SysFile.id == SysItemFile.file_id)
            .where(SysDatasetItem.dataset_id.in_(dataset_ids))
            .group_by(SysDatasetItem.dataset_id, SysFile.type)
        )
        result = await db.execute(stmt)
        dist_map: dict[int, dict[str, int]] = {}
        for row in result:
            ds_id = int(row.dataset_id)
            key = str(row.key or "unknown")
            if ds_id not in dist_map:
                dist_map[ds_id] = {}
            dist_map[ds_id][key] = int(row.cnt)
        return dist_map

    async def _count_distribution_batch(
        self,
        db: AsyncSession,
        dataset_ids: list[int],
        column,
        default_label: str,
    ) -> dict[int, dict[str, int]]:
        """通用批量分布统计"""
        if not dataset_ids:
            return {}
        stmt = (
            select(
                SysDatasetItem.dataset_id.label("dataset_id"),
                func.coalesce(column, default_label).label("key"),
                func.count(SysItemFile.id).label("cnt"),
            )
            .select_from(SysDatasetItem)
            .join(SysItemFile, SysItemFile.item_id == SysDatasetItem.id)
            .where(SysDatasetItem.dataset_id.in_(dataset_ids))
            .group_by(SysDatasetItem.dataset_id, func.coalesce(column, default_label))
        )
        result = await db.execute(stmt)
        dist_map: dict[int, dict[str, int]] = {}
        for row in result:
            ds_id = int(row.dataset_id)
            key = str(row.key or default_label)
            if ds_id not in dist_map:
                dist_map[ds_id] = {}
            dist_map[ds_id][key] = int(row.cnt)
        return dist_map

    async def get_items_with_files_batch(
        self,
        db: AsyncSession,
        item_ids: list[int],
    ) -> tuple[dict[int, SysDatasetItem], dict[int, list[tuple[SysItemFile, SysFile]]]]:
        """批量查询数据项及其关联文件（避免N+1）"""
        if not item_ids:
            return {}, {}

        items_stmt = select(SysDatasetItem).where(SysDatasetItem.id.in_(item_ids))
        items_result = await db.execute(items_stmt)
        items_map: dict[int, SysDatasetItem] = {}
        for item in items_result.scalars().all():
            items_map[int(item.id)] = item

        files_stmt = (
            select(SysItemFile, SysFile)
            .select_from(SysItemFile)
            .join(SysFile, SysItemFile.file_id == SysFile.id)
            .where(SysItemFile.item_id.in_(item_ids))
        )
        files_result = await db.execute(files_stmt)
        files_map: dict[int, list[tuple[SysItemFile, SysFile]]] = {}
        for row in files_result.all():
            item_file = row[0]
            file_obj = row[1]
            iid = int(item_file.item_id)
            if iid not in files_map:
                files_map[iid] = []
            files_map[iid].append((item_file, file_obj))

        return items_map, files_map


# 单例
dataset_repository = DatasetRepository()
