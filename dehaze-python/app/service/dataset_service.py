"""
数据集服务模块 - 基于事件驱动架构的重构实现
参考 dehaze-java 的数据集模块重构设计
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
import re
import json
from collections import deque

from app.extensions import mysql, redis_client
from app.models import (
    SysDataset, SysDatasetItem, SysItemFile, SysFile,
    DatasetAddForm, DatasetUpdateForm, DatasetQuery,
    DatasetItemCreateForm, DatasetItemUpdateForm, DatasetItemUploadForm,
    BatchDatasetItemUploadForm, ItemFileUpdateForm,
    DatasetVO, DatasetItemVO, ItemFileVO, DatasetStatistics,
    BatchDeleteResult, BatchDeleteResultItem, BatchOperationResultVO,
    BatchUploadResultVO, BatchUploadSuccessItemVO, BatchUploadFailedItemVO,
    BatchActionFailureDetailVO, ImageType, HazeLevel
)
from sqlalchemy import func, and_, or_
from sqlalchemy.orm import joinedload


class DatasetService:
    """数据集服务类 - 支持缓存和性能优化"""

    # 缓存键前缀
    CACHE_TREE_PREFIX = "dataset:tree:"
    CACHE_STATS_PREFIX = "dataset:stats:"
    CACHE_TTL = 3600  # 缓存1小时

    # 根节点ID
    ROOT_DATASET_ID = 0

    @staticmethod
    def get_dataset_tree(query: DatasetQuery = None) -> List[Dict[str, Any]]:
        """
        获取数据集树（带缓存优化）

        Args:
            query: 查询条件

        Returns:
            数据集树结构（带统计信息）
        """
        cache_key = f"{DatasetService.CACHE_TREE_PREFIX}all"

        # 尝试从缓存获取
        cached_data = redis_client.get(cache_key)
        if cached_data:
            try:
                return json.loads(cached_data)
            except (json.JSONDecodeError, Exception):
                pass

        # 构建查询
        db_query = SysDataset.query.filter(SysDataset.deleted == 0)

        if query and query.keyword:
            db_query = db_query.filter(SysDataset.name.contains(query.keyword))

        datasets = db_query.all()

        # 使用 BFS 一次性加载所有节点，优化性能
        tree = DatasetService._build_tree_with_bfs(0, datasets)

        # 缓存结果
        try:
            redis_client.setex(cache_key, DatasetService.CACHE_TTL,
                               json.dumps(tree, ensure_ascii=False))
        except Exception:
            pass

        return tree

    @staticmethod
    def _build_tree_with_bfs(parent_id: int, datasets: List[SysDataset]) -> List[Dict[str, Any]]:
        """
        使用 BFS 算法构建数据集树（优化 N+1 查询问题）

        Args:
            parent_id: 父节点ID
            datasets: 所有数据集列表

        Returns:
            树形结构列表
        """
        # 构建ID到节点的映射
        dataset_map = {ds.id: ds for ds in datasets}

        # 构建父子关系映射
        children_map: Dict[int, List[SysDataset]] = {}
        for dataset in datasets:
            if dataset.parent_id not in children_map:
                children_map[dataset.parent_id] = []
            children_map[dataset.parent_id].append(dataset)

        # 使用 BFS 遍历
        tree = []
        queue = deque()

        # 处理根节点
        root_children = children_map.get(0, [])
        for root_child in root_children:
            queue.append((root_child, tree, True))

        while queue:
            dataset, parent_list, is_root_child = queue.popleft()

            # 获取或计算统计信息
            statistics = DatasetService._get_or_calculate_statistics(dataset.id)

            # 构建 VO
            vo = DatasetVO(dataset, statistics)
            vo_dict = vo.to_dict()

            if is_root_child:
                parent_list.append(vo_dict)
                vo_dict['children'] = []
                queue.extend([(child, vo_dict['children'], False)
                              for child in children_map.get(dataset.id, [])])

            # 添加子节点
            children = children_map.get(dataset.id, [])
            if children:
                vo_dict['children'] = []
                for child in children:
                    child_stats = DatasetService._get_or_calculate_statistics(child.id)
                    child_vo = DatasetVO(child, child_stats)
                    child_dict = child_vo.to_dict()
                    vo_dict['children'].append(child_dict)
                    # 将子节点的子节点加入队列
                    queue.extend([(grandchild, child_dict['children'], False)
                                  for grandchild in children_map.get(child.id, [])])

        return tree

    @staticmethod
    def _get_or_calculate_statistics(dataset_id: int) -> Optional[DatasetStatistics]:
        """
        获取或计算数据集统计信息（带缓存）

        Args:
            dataset_id: 数据集ID

        Returns:
            统计信息对象
        """
        if dataset_id == 0:
            return None

        cache_key = f"{DatasetService.CACHE_STATS_PREFIX}{dataset_id}"

        # 尝试从缓存获取
        cached_data = redis_client.get(cache_key)
        if cached_data:
            try:
                data = json.loads(cached_data)
                stats = DatasetStatistics()
                stats.item_count = data.get('itemCount', 0)
                stats.file_count = data.get('fileCount', 0)
                stats.total_size = data.get('totalSize', 0)
                stats.clear_count = data.get('clearCount', 0)
                stats.hazy_count = data.get('hazyCount', 0)
                stats.scene_distribution = data.get('sceneDistribution', {})
                stats.haze_distribution = data.get('hazeDistribution', {})
                stats.format_distribution = data.get('formatDistribution', {})
                return stats
            except Exception:
                pass

        # 计算统计信息
        leaf_ids = DatasetService._get_leaf_dataset_ids(dataset_id)
        stats = DatasetService._calculate_statistics_for_datasets(leaf_ids)

        # 缓存结果
        try:
            redis_client.setex(cache_key, DatasetService.CACHE_TTL,
                               json.dumps(stats.to_dict(), ensure_ascii=False))
        except Exception:
            pass

        return stats

    @staticmethod
    def _calculate_statistics_for_datasets(dataset_ids: List[int]) -> DatasetStatistics:
        """
        计算指定数据集的统计信息（使用 SQL 聚合查询）

        Args:
            dataset_ids: 数据集ID列表（叶子节点）

        Returns:
            统计信息对象
        """
        stats = DatasetStatistics()

        if not dataset_ids:
            return stats

        # 统计数据项总数
        stats.item_count = mysql.session.query(func.count(SysDatasetItem.id)) \
                               .filter(SysDatasetItem.dataset_id.in_(dataset_ids)) \
                               .scalar() or 0

        # 统计文件总数和总大小
        query = mysql.session.query(
            func.count(SysItemFile.id),
            func.sum(SysFile.size)
        ).join(
            SysFile, SysItemFile.file_id == SysFile.id
        ).filter(
            SysItemFile.item_id.in_(
                mysql.session.query(SysDatasetItem.id)
                .filter(SysDatasetItem.dataset_id.in_(dataset_ids))
            )
        ).first()

        stats.file_count = query[0] or 0
        stats.total_size = query[1] or 0

        # 统计清晰图和有雾图数量
        clear_count = mysql.session.query(func.count(SysItemFile.id)) \
                          .join(
            SysDatasetItem, SysItemFile.item_id == SysDatasetItem.id
        ).filter(
            and_(
                SysDatasetItem.dataset_id.in_(dataset_ids),
                or_(
                    func.lower(SysItemFile.type).like('%clear%'),
                    func.lower(SysItemFile.type).like('%clean%'),
                    SysItemFile.type.like('%清晰%'),
                    SysItemFile.type.like('%无雾%')
                )
            )
        ).scalar() or 0

        hazy_count = mysql.session.query(func.count(SysItemFile.id)) \
                         .join(
            SysDatasetItem, SysItemFile.item_id == SysDatasetItem.id
        ).filter(
            and_(
                SysDatasetItem.dataset_id.in_(dataset_ids),
                or_(
                    func.lower(SysItemFile.type).like('%haze%'),
                    func.lower(SysItemFile.type).like('%hazy%'),
                    SysItemFile.type.like('%有雾%')
                )
            )
        ).scalar() or 0

        stats.clear_count = clear_count
        stats.hazy_count = hazy_count

        # 统计场景类型分布
        scene_query = mysql.session.query(
            func.coalesce(SysItemFile.scene_type, '未分类').label('scene_type'),
            func.count(SysItemFile.id).label('count')
        ).join(
            SysDatasetItem, SysItemFile.item_id == SysDatasetItem.id
        ).filter(
            SysDatasetItem.dataset_id.in_(dataset_ids)
        ).group_by(
            func.coalesce(SysItemFile.scene_type, '未分类')
        ).all()

        stats.scene_distribution = {row.scene_type: row.count for row in scene_query}

        # 统计雾霾程度分布
        haze_query = mysql.session.query(
            func.coalesce(SysItemFile.haze_level, '未标注').label('haze_level'),
            func.count(SysItemFile.id).label('count')
        ).join(
            SysDatasetItem, SysItemFile.item_id == SysDatasetItem.id
        ).filter(
            SysDatasetItem.dataset_id.in_(dataset_ids)
        ).group_by(
            func.coalesce(SysItemFile.haze_level, '未标注')
        ).all()

        stats.haze_distribution = {row.haze_level: row.count for row in haze_query}

        # 统计文件格式分布
        format_query = mysql.session.query(
            SysFile.type.label('file_type'),
            func.count(SysFile.id).label('count')
        ).join(
            SysItemFile, SysFile.id == SysItemFile.file_id
        ).join(
            SysDatasetItem, SysItemFile.item_id == SysDatasetItem.id
        ).filter(
            SysDatasetItem.dataset_id.in_(dataset_ids)
        ).group_by(
            SysFile.type
        ).all()

        stats.format_distribution = {row.file_type: row.count for row in format_query}

        return stats

    @staticmethod
    def _get_leaf_dataset_ids(dataset_id: int) -> List[int]:
        """
        获取指定节点下的所有叶子节点ID（使用 BFS 优化）

        Args:
            dataset_id: 数据集ID

        Returns:
            叶子节点ID列表
        """
        # 获取所有数据集
        all_datasets = SysDataset.query.filter(SysDataset.deleted == 0).all()

        # 构建父子关系映射
        children_map: Dict[int, List[SysDataset]] = {}
        for dataset in all_datasets:
            if dataset.parent_id not in children_map:
                children_map[dataset.parent_id] = []
            children_map[dataset.parent_id].append(dataset)

        # 使用 BFS 获取所有子节点
        queue = deque([dataset_id])
        all_ids = [dataset_id]

        while queue:
            current_id = queue.popleft()
            children = children_map.get(current_id, [])
            for child in children:
                all_ids.append(child.id)
                queue.append(child.id)

        # 找出叶子节点（没有子节点的节点）
        leaf_ids = [ds_id for ds_id in all_ids if not children_map.get(ds_id)]

        return leaf_ids if leaf_ids else [dataset_id]

    @staticmethod
    def get_dataset_by_id(dataset_id: int) -> Optional[DatasetVO]:
        """
        根据ID获取数据集详情

        Args:
            dataset_id: 数据集ID

        Returns:
            数据集VO对象
        """
        dataset = SysDataset.query.filter(
            and_(SysDataset.id == dataset_id, SysDataset.deleted == 0)
        ).first()

        if not dataset:
            return None

        statistics = DatasetService._get_or_calculate_statistics(dataset_id)
        return DatasetVO(dataset, statistics)

    @staticmethod
    def create_dataset(form: DatasetAddForm) -> DatasetVO:
        """
        创建数据集

        Args:
            form: 数据集添加表单

        Returns:
            创建的数据集VO
        """
        # 验证父数据集是否存在
        if form.parent_id != 0:
            parent = SysDataset.query.filter(
                and_(SysDataset.id == form.parent_id, SysDataset.deleted == 0)
            ).first()
            if not parent:
                raise ValueError("父数据集不存在")

        # 生成树路径
        tree_path = DatasetService._generate_tree_path(form.parent_id)

        # 创建数据集
        dataset = SysDataset()
        dataset.parent_id = form.parent_id
        dataset.tree_path = tree_path
        dataset.type = form.type
        dataset.name = form.name
        dataset.description = form.description
        dataset.path = form.path
        dataset.status = form.status
        dataset.deleted = 0
        dataset.create_time = datetime.now()
        dataset.update_time = datetime.now()

        mysql.session.add(dataset)
        mysql.session.commit()

        # 清除父节点及其祖先的缓存
        if form.parent_id != 0:
            DatasetService._evict_dataset_and_ancestor_stats_cache(form.parent_id)
        DatasetService._evict_tree_cache()

        return DatasetService.get_dataset_by_id(dataset.id)

    @staticmethod
    def _generate_tree_path(parent_id: int) -> str:
        """
        生成树路径

        Args:
            parent_id: 父节点ID

        Returns:
            树路径字符串
        """
        if parent_id == 0:
            return "0"

        # 获取父数据集的树路径
        parent = SysDataset.query.get(parent_id)
        if not parent or not parent.tree_path:
            return f"0,{parent_id}"

        return f"{parent.tree_path},{parent_id}"

    @staticmethod
    def update_dataset(dataset_id: int, form: DatasetUpdateForm) -> DatasetVO:
        """
        更新数据集

        Args:
            dataset_id: 数据集ID
            form: 数据集更新表单

        Returns:
            更新后的数据集VO
        """
        dataset = SysDataset.query.filter(
            and_(SysDataset.id == dataset_id, SysDataset.deleted == 0)
        ).first()

        if not dataset:
            raise ValueError("数据集不存在")

        old_parent_id = dataset.parent_id

        # 更新字段
        if form.parent_id is not None and form.parent_id != old_parent_id:
            # 验证新父数据集
            if form.parent_id != 0:
                new_parent = SysDataset.query.filter(
                    and_(SysDataset.id == form.parent_id, SysDataset.deleted == 0)
                ).first()
                if not new_parent:
                    raise ValueError("新父数据集不存在")

            # 防止循环引用
            if DatasetService._would_create_cycle(dataset_id, form.parent_id):
                raise ValueError("不能将数据集移动到其子节点下")

            # 更新树路径
            old_tree_path = dataset.tree_path
            new_tree_path = DatasetService._generate_tree_path(form.parent_id)
            dataset.tree_path = new_tree_path
            dataset.parent_id = form.parent_id

            # 更新所有子节点的树路径
            DatasetService._update_children_tree_paths(dataset_id, old_tree_path, new_tree_path)

            # 清除缓存
            DatasetService._evict_dataset_stats_cache(dataset_id)
            if old_parent_id != 0:
                DatasetService._evict_dataset_and_ancestor_stats_cache(old_parent_id)
            if form.parent_id != 0:
                DatasetService._evict_dataset_and_ancestor_stats_cache(form.parent_id)

        if form.name is not None:
            dataset.name = form.name
        if form.type is not None:
            dataset.type = form.type
        if form.description is not None:
            dataset.description = form.description
        if form.path is not None:
            dataset.path = form.path
        if form.status is not None:
            dataset.status = form.status

        dataset.update_time = datetime.now()

        mysql.session.commit()

        DatasetService._evict_tree_cache()

        return DatasetService.get_dataset_by_id(dataset_id)

    @staticmethod
    def _would_create_cycle(dataset_id: int, new_parent_id: int) -> bool:
        """
        检查是否会产生循环引用

        Args:
            dataset_id: 当前数据集ID
            new_parent_id: 新父节点ID

        Returns:
            是否会产生循环引用
        """
        if new_parent_id == 0:
            return False

        # 检查新父节点是否是当前节点的后代
        descendants = DatasetService._get_all_descendant_ids(dataset_id)
        return new_parent_id in descendants

    @staticmethod
    def _get_all_descendant_ids(dataset_id: int) -> List[int]:
        """
        获取所有后代节点ID

        Args:
            dataset_id: 数据集ID

        Returns:
            后代节点ID列表
        """
        all_datasets = SysDataset.query.filter(SysDataset.deleted == 0).all()
        children_map: Dict[int, List[SysDataset]] = {}

        for dataset in all_datasets:
            if dataset.parent_id not in children_map:
                children_map[dataset.parent_id] = []
            children_map[dataset.parent_id].append(dataset)

        queue = deque([dataset_id])
        descendant_ids = []

        while queue:
            current_id = queue.popleft()
            children = children_map.get(current_id, [])
            for child in children:
                descendant_ids.append(child.id)
                queue.append(child.id)

        return descendant_ids

    @staticmethod
    def _update_children_tree_paths(dataset_id: int, old_prefix: str, new_prefix: str):
        """
        更新所有子节点的树路径

        Args:
            dataset_id: 数据集ID
            old_prefix: 旧路径前缀
            new_prefix: 新路径前缀
        """
        children = DatasetService._get_all_descendant_ids(dataset_id)

        for child_id in children:
            child = SysDataset.query.get(child_id)
            if child and child.tree_path and child.tree_path.startswith(old_prefix):
                suffix = child.tree_path[len(old_prefix):]
                child.tree_path = f"{new_prefix}{suffix}"

        mysql.session.commit()

    @staticmethod
    def batch_delete_datasets(dataset_ids: List[int]) -> BatchDeleteResult:
        """
        批量删除数据集（级联删除子数据集、数据项、文件）

        Args:
            dataset_ids: 数据集ID列表

        Returns:
            批量删除结果
        """
        total = len(dataset_ids)
        succeeded = 0
        failed = 0
        results: List[BatchDeleteResultItem] = []

        for dataset_id in dataset_ids:
            try:
                # 获取父数据集ID
                dataset = SysDataset.query.get(dataset_id)
                parent_id = dataset.parent_id if dataset else None

                # 获取所有需要删除的数据集ID（包括子节点）
                all_dataset_ids = DatasetService._get_dataset_and_descendant_ids(dataset_id)

                # 获取所有叶子节点数据集
                leaf_ids = []
                all_datasets = SysDataset.query.filter(SysDataset.id.in_(all_dataset_ids)).all()
                children_map: Dict[int, List[SysDataset]] = {}
                for ds in all_datasets:
                    if ds.parent_id not in children_map:
                        children_map[ds.parent_id] = []
                    children_map[ds.parent_id].append(ds)

                for ds_id in all_dataset_ids:
                    if not children_map.get(ds_id):
                        leaf_ids.append(ds_id)

                # 删除所有数据项（包括关联的文件）
                for leaf_id in leaf_ids:
                    items = SysDatasetItem.query.filter(
                        SysDatasetItem.dataset_id == leaf_id
                    ).all()
                    for item in items:
                        DatasetItemService.delete_item_cascade(item.id)

                # 从叶子节点往上删除数据集
                all_dataset_ids.sort(reverse=True, key=lambda x: DatasetService._get_tree_depth(x))
                for ds_id in all_dataset_ids:
                    SysDataset.query.filter(SysDataset.id == ds_id).delete()

                succeeded += 1

                # 清除已删除数据集的缓存
                for deleted_id in all_dataset_ids:
                    DatasetService._evict_dataset_stats_cache(deleted_id)
                if parent_id and parent_id != 0:
                    DatasetService._evict_dataset_and_ancestor_stats_cache(parent_id)

                results.append(BatchDeleteResultItem(
                    dataset_id=dataset_id,
                    status="success"
                ))

            except Exception as e:
                failed += 1
                results.append(BatchDeleteResultItem(
                    dataset_id=dataset_id,
                    status="failed",
                    message=str(e),
                    error_code="SYSTEM_ERROR"
                ))

        mysql.session.commit()
        DatasetService._evict_tree_cache()

        return BatchDeleteResult(
            total=total,
            succeeded=succeeded,
            failed=failed,
            results=results
        )

    @staticmethod
    def _get_dataset_and_descendant_ids(dataset_id: int) -> List[int]:
        """
        获取数据集及其所有后代ID

        Args:
            dataset_id: 数据集ID

        Returns:
            数据集ID列表（包括自己和所有子节点）
        """
        return [dataset_id] + DatasetService._get_all_descendant_ids(dataset_id)

    @staticmethod
    def _get_tree_depth(dataset_id: int) -> int:
        """
        计算节点在树中的深度

        Args:
            dataset_id: 数据集ID

        Returns:
            树深度
        """
        dataset = SysDataset.query.get(dataset_id)
        if not dataset or not dataset.tree_path:
            return 0
        return dataset.tree_path.count(',')

    @staticmethod
    def _evict_dataset_stats_cache(dataset_id: int):
        """
        清除指定数据集的统计缓存

        Args:
            dataset_id: 数据集ID
        """
        cache_key = f"{DatasetService.CACHE_STATS_PREFIX}{dataset_id}"
        try:
            redis_client.delete(cache_key)
        except Exception:
            pass

    @staticmethod
    def _evict_dataset_and_ancestor_stats_cache(dataset_id: int):
        """
        清除数据集及其祖先的统计缓存

        Args:
            dataset_id: 数据集ID
        """
        dataset = SysDataset.query.get(dataset_id)
        if not dataset or not dataset.tree_path:
            return

        # tree_path 格式为 "0,1,2,3"，需要反转为 [3, 2, 1, 0]
        ancestor_ids = [int(x) for x in reversed(dataset.tree_path.split(','))]

        for ancestor_id in ancestor_ids:
            DatasetService._evict_dataset_stats_cache(ancestor_id)

    @staticmethod
    def _evict_tree_cache():
        """清除数据集树缓存"""
        cache_key = f"{DatasetService.CACHE_TREE_PREFIX}all"
        try:
            redis_client.delete(cache_key)
        except Exception:
            pass

    @staticmethod
    def get_leaf_dataset_ids(dataset_id: int) -> List[int]:
        """
        获取指定数据集下的所有叶子节点ID

        Args:
            dataset_id: 数据集ID

        Returns:
            叶子节点ID列表
        """
        return DatasetService._get_leaf_dataset_ids(dataset_id)

    @staticmethod
    def evict_dataset_stats_cache(dataset_id: int):
        """
        公开方法：清除指定数据集的统计缓存
        供其他服务调用

        Args:
            dataset_id: 数据集ID
        """
        DatasetService._evict_dataset_stats_cache(dataset_id)

    @staticmethod
    def evict_dataset_and_ancestor_stats_cache(dataset_id: int):
        """
        公开方法：清除数据集及其祖先的统计缓存
        供其他服务调用

        Args:
            dataset_id: 数据集ID
        """
        DatasetService._evict_dataset_and_ancestor_stats_cache(dataset_id)


class DatasetItemService:
    """数据集项服务类"""

    @staticmethod
    def create_dataset_item(form: DatasetItemCreateForm) -> DatasetItemVO:
        """
        创建数据集项

        Args:
            form: 数据集项创建表单

        Returns:
            创建的数据集项VO
        """
        # 验证数据集存在且不是目录
        dataset = SysDataset.query.filter(
            and_(SysDataset.id == form.dataset_id, SysDataset.deleted == 0)
        ).first()

        if not dataset:
            raise ValueError("数据集不存在")

        # 检查是否是叶子节点
        children_count = SysDataset.query.filter(
            SysDataset.parent_id == form.dataset_id
        ).count()

        if children_count > 0:
            raise ValueError("不能在目录类型的数据集中创建数据项")

        dataset_item = SysDatasetItem()
        dataset_item.dataset_id = form.dataset_id
        dataset_item.name = form.name
        dataset_item.create_time = datetime.now()
        dataset_item.update_time = datetime.now()

        mysql.session.add(dataset_item)
        mysql.session.commit()

        # 清除数据集统计缓存
        DatasetService.evict_dataset_stats_cache(form.dataset_id)

        return DatasetItemService.get_dataset_item(dataset_item.id)

    @staticmethod
    def get_dataset_item(item_id: int) -> DatasetItemVO:
        """
        获取数据集项详情

        Args:
            item_id: 数据集项ID

        Returns:
            数据集项VO
        """
        item = SysDatasetItem.query.get(item_id)
        if not item:
            raise ValueError("数据项不存在")

        # 获取关联的文件
        item_files = mysql.session.query(SysItemFile).join(
            SysFile, SysItemFile.file_id == SysFile.id
        ).join(
            SysDatasetItem, SysItemFile.item_id == SysDatasetItem.id
        ).filter(
            SysItemFile.item_id == item_id
        ).all()

        files = []
        image_urls = []

        for item_file in item_files:
            file_obj = mysql.session.query(SysFile).get(item_file.file_id)
            if file_obj:
                files.append(ItemFileVO(item_file, file_obj).__dict__)
                image_urls.append({
                    'id': file_obj.id,
                    'type': item_file.type,
                    'url': file_obj.url,
                    'thumbnailUrl': file_obj.url  # 可以后续添加缩略图处理
                })

        vo = DatasetItemVO(item, files)
        vo.image_urls = image_urls
        return vo

    @staticmethod
    def update_dataset_item(item_id: int, form: DatasetItemUpdateForm):
        """
        更新数据集项

        Args:
            item_id: 数据集项ID
            form: 数据集项更新表单
        """
        item = SysDatasetItem.query.get(item_id)
        if not item:
            raise ValueError("数据项不存在")

        item.name = form.name
        item.update_time = datetime.now()

        mysql.session.commit()

    @staticmethod
    def delete_item_cascade(item_id: int):
        """
        级联删除数据项（包括关联的文件记录）

        Args:
            item_id: 数据集项ID
        """
        item = SysDatasetItem.query.get(item_id)
        if not item:
            raise ValueError("数据项不存在")

        dataset_id = item.dataset_id

        # 删除关联的文件项记录
        SysItemFile.query.filter(SysItemFile.item_id == item_id).delete()

        # 删除数据项
        SysDatasetItem.query.filter(SysDatasetItem.id == item_id).delete()

        mysql.session.commit()

        # 清除数据集统计缓存
        DatasetService.evict_dataset_stats_cache(dataset_id)

    @staticmethod
    def batch_delete_items_cascade(item_ids: List[int]) -> BatchOperationResultVO:
        """
        批量级联删除数据项

        Args:
            item_ids: 数据集项ID列表

        Returns:
            批量操作结果
        """
        success_count = 0
        failed_count = 0
        success_ids = []
        failure_details = []

        for item_id in item_ids:
            try:
                DatasetItemService.delete_item_cascade(item_id)
                success_count += 1
                success_ids.append(item_id)
            except Exception as e:
                failed_count += 1
                failure_details.append(BatchActionFailureDetailVO(
                    identifier=str(item_id),
                    reason=str(e)
                ))

        return BatchOperationResultVO(
            success_count=success_count,
            failed_count=failed_count,
            success_ids=success_ids,
            failure_details=failure_details,
            message=f"批量删除完成：成功{success_count}个，失败{failed_count}个"
        )

    @staticmethod
    def get_items_by_dataset(dataset_id: int, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """
        分页获取数据集项

        Args:
            dataset_id: 数据集ID
            page: 页码
            page_size: 每页数量

        Returns:
            分页结果
        """
        # 获取叶子节点
        leaf_ids = DatasetService._get_leaf_dataset_ids(dataset_id)

        # 分页查询
        query = SysDatasetItem.query.filter(
            SysDatasetItem.dataset_id.in_(leaf_ids)
        )

        total = query.count()
        items = query.offset((page - 1) * page_size).limit(page_size).all()

        # 构建返回数据
        records = []
        for item in items:
            item_vo = DatasetItemService.get_dataset_item(item.id)
            records.append(item_vo.to_dict())

        return {
            'records': records,
            'total': total,
            'page': page,
            'pageSize': page_size
        }


class DatasetOperationService:
    """数据集操作服务 - 处理跨服务的复杂组合操作"""

    @staticmethod
    def create_item_with_images(form: DatasetItemUploadForm) -> DatasetItemVO:
        """
        创建数据项并上传配对图片

        Args:
            form: 配对上传表单

        Returns:
            创建的数据项VO
        """
        # 校验配对图片分辨率一致性
        DatasetOperationService._validate_paired_image_resolution(form)

        # 创建数据项
        items_form = DatasetItemCreate(form.dataset_id, form.name)
        dataset_item = DatasetItemService.create_dataset_item(items_form)

        # 保存文件（这里需要调用文件服务）
        # TODO: 集成文件服务后完成

        return DatasetItemService.get_dataset_item(dataset_item.id)

    @staticmethod
    def _validate_paired_image_resolution(form: DatasetItemUploadForm):
        """
        校验配对图片分辨率一致性

        Args:
            form: 上传表单
        """
        # TODO: 实现图片分辨率校验
        # 需要Pillow库来解析图片尺寸
        pass

    @staticmethod
    def batch_create_items_with_images(form: BatchDatasetItemUploadForm) -> BatchUploadResultVO:
        """
        批量创建数据项并上传配对图片

        Args:
            form: 批量上传表单

        Returns:
            批量处理结果
        """
        # 按文件名前缀分组
        file_groups = DatasetOperationService._group_files_by_prefix(form.files)

        success_groups = 0
        failed_groups = 0
        success_items = []
        failed_items = []

        for group_name, group_files in file_groups.items():
            try:
                # 验证组完整性
                if "clear" not in group_files:
                    raise ValueError("缺少清晰图（需包含_clear或_gt后缀）")
                if "hazy" not in group_files or not group_files["hazy"]:
                    raise ValueError("缺少有雾图（需包含_hazy后缀）")

                # 创建单个表单
                single_form = DatasetItemUploadForm(
                    dataset_id=form.dataset_id,
                    clear_image=group_files["clear"],
                    hazy_images=[f["file"] for f in group_files["hazy"]],
                    haze_levels=[f["hazeLevel"] for f in group_files["hazy"]],
                    name=group_name,
                    scene_type=form.scene_type
                )

                created_item = DatasetOperationService.create_item_with_images(single_form)
                success_groups += 1

                file_count = 1 + len(group_files["hazy"])
                success_items.append(BatchUploadSuccessItemVO(
                    dataset_item_id=created_item.id,
                    name=group_name,
                    file_count=file_count
                ))

            except Exception as e:
                failed_groups += 1
                failed_items.append(BatchUploadFailedItemVO(
                    filename=group_name,
                    error_message=str(e)
                ))

        return BatchUploadResultVO(
            total=form.files.__len__() if form.files else 0,
            success=success_groups,
            failed=failed_groups,
            success_items=success_items,
            failed_items=failed_items
        )

    @staticmethod
    def _group_files_by_prefix(files: List) -> Dict[str, Dict[str, Any]]:
        """
        按文件名前缀分组文件

        Args:
            files: 文件列表

        Returns:
            分组后的文件字典
        """
        groups = {}

        for file in files:
            filename = getattr(file, 'filename', '')
            if not filename:
                continue

            prefix = DatasetOperationService._extract_file_prefix(filename)

            if prefix not in groups:
                groups[prefix] = {}

            if DatasetOperationService._is_clear_image(filename):
                groups[prefix]["clear"] = file
            elif DatasetOperationService._is_hazy_image(filename):
                try:
                    haze_level = DatasetOperationService._extract_haze_level(filename)
                    if "hazy" not in groups[prefix]:
                        groups[prefix]["hazy"] = []
                    groups[prefix]["hazy"].append({
                        "file": file,
                        "hazeLevel": haze_level
                    })
                except Exception:
                    pass

        return groups

    @staticmethod
    def _extract_file_prefix(filename: str) -> str:
        """
        从文件名提取前缀

        Args:
            filename: 文件名

        Returns:
            文件前缀
        """
        name_without_ext = filename.rsplit('.', 1)[0]
        return re.sub(r'(_clear|_gt|_hazy.*)$', '', name_without_ext)

    @staticmethod
    def _is_clear_image(filename: str) -> bool:
        """
        判断是否为清晰图

        Args:
            filename: 文件名

        Returns:
            是否为清晰图
        """
        return '_clear' in filename.lower() or '_gt' in filename.lower()

    @staticmethod
    def _is_hazy_image(filename: str) -> bool:
        """
        判断是否为有雾图

        Args:
            filename: 文件名

        Returns:
            是否为有雾图
        """
        return '_hazy' in filename.lower()

    @staticmethod
    def _extract_haze_level(filename: str) -> str:
        """
        从文件名提取雾霾程度

        Args:
            filename: 文件名

        Returns:
            雾霾程度 (light/medium/heavy)

        Raises:
            ValueError: 无法提取雾霾程度
        """
        pattern = r'.*_hazy_(light|medium|heavy).*'
        match = re.fullmatch(pattern, filename.lower())

        if match:
            return match.group(1)

        raise ValueError("文件名必须包含雾霾程度标识(light/medium/heavy)")

    @staticmethod
    def batch_delete_items_cascade_with_result(item_ids: List[int]) -> BatchOperationResultVO:
        """
        批量级联删除数据项（带详细结果）

        Args:
            item_ids: 数据集项ID列表

        Returns:
            批量操作结果
        """
        return DatasetItemService.batch_delete_items_cascade(item_ids)

    @staticmethod
    def batch_delete_datasets(dataset_ids: List[int]) -> BatchDeleteResult:
        """
        级联删除数据集（包括子数据集、数据项、文件）

        Args:
            dataset_ids: 数据集ID列表

        Returns:
            批量删除结果
        """
        return DatasetService.batch_delete_datasets(dataset_ids)
