"""
数据集服务

提供数据集 CRUD 功能，支持树形结构、数据项管理
"""

import json
import logging
import re
from datetime import datetime
from typing import Any

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_dataset import (SysDataset, SysDatasetItem,
                                           SysItemFile)
from app.repository.dataset_repository import dataset_repository
from app.service.file_service import FileService
from app.utils.datetime_utils import format_time
from app.utils.tree import generate_tree_path
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


def _extract_file_prefix(filename: str) -> str:
    """提取文件名前缀（去除 _clear/_gt/_hazy_* 后缀和扩展名）"""
    name = re.sub(r'\.[^.]+$', '', filename)
    name = re.sub(r'_(clear|gt|hazy.*)$', '', name, flags=re.IGNORECASE)
    return name


def _is_clear_image(filename: str) -> bool:
    """判断文件名是否为清晰图（含 _clear 或 _gt）"""
    return bool(re.search(r'_(clear|gt)\b', filename, re.IGNORECASE))


def _is_hazy_image(filename: str) -> bool:
    """判断文件名是否为有雾图（含 _hazy）"""
    return '_hazy' in filename.lower()


def _extract_haze_level(filename: str) -> str:
    """从文件名提取雾霾程度，默认 medium"""
    match = re.search(r'_hazy_(light|medium|heavy)', filename, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "medium"


class DatasetService:
    """数据集服务（异步版本）"""

    # 缓存键前缀
    CACHE_TREE_KEY = "dataset:tree"
    CACHE_STATS_PREFIX = "dataset:stats:"
    CACHE_TREE_TTL = 3600  # 树缓存1小时
    CACHE_STATS_TTL = 1800  # 统计缓存30分钟

    # 根节点ID
    ROOT_DATASET_ID = 0

    @staticmethod
    async def get_dataset_tree(
        db: AsyncSession,
        redis: Redis,
        keywords: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        获取数据集树（带缓存优化）

        Args:
            db: 数据库会话
            redis: Redis 客户端
            keywords: 搜索关键字

        Returns:
            数据集树结构（带统计信息）
        """
        # 仅无关键字时使用树缓存
        use_cache = not keywords

        if use_cache:
            try:
                cached_data = await redis.get(DatasetService.CACHE_TREE_KEY)
                if cached_data:
                    return json.loads(cached_data)
            except Exception:
                pass

        # 使用 Repository 查询
        datasets = await dataset_repository.get_list_with_keywords(db, keywords)

        # 使用 BFS 一次性加载所有节点，优化性能
        tree = await DatasetService._build_tree_with_bfs(db, redis, datasets)

        # 仅无关键字时写入缓存
        if use_cache:
            try:
                await redis.setex(
                    DatasetService.CACHE_TREE_KEY,
                    DatasetService.CACHE_TREE_TTL,
                    json.dumps(tree, ensure_ascii=False),
                )
            except Exception as e:
                logger.warning(f"缓存写入失败: {e}")

        return tree

    @staticmethod
    async def _build_tree_with_bfs(
        db: AsyncSession,
        redis: Redis,
        datasets: list[SysDataset],
    ) -> list[dict[str, Any]]:
        """
        使用 BFS 算法构建数据集树（优化 N+1 查询问题）

        Args:
            db: 数据库会话
            redis: Redis 客户端
            datasets: 所有数据集列表

        Returns:
            树形结构列表
        """
        # 构建父子关系映射
        children_map: dict[int, list[SysDataset]] = {}
        for dataset in datasets:
            if dataset.parent_id not in children_map:
                children_map[dataset.parent_id] = []
            children_map[dataset.parent_id].append(dataset)

        # 使用 BFS 遍历构建树
        tree = []

        # 处理根节点
        root_children = children_map.get(0, [])
        for root_child in root_children:
            node_dict = await DatasetService._build_dataset_node(db, redis, root_child, children_map)
            tree.append(node_dict)

        return tree

    @staticmethod
    async def _build_dataset_node(
        db: AsyncSession,
        redis: Redis,
        dataset: SysDataset,
        children_map: dict[int, list[SysDataset]],
    ) -> dict[str, Any]:
        """
        递归构建数据集节点

        Args:
            db: 数据库会话
            redis: Redis 客户端
            dataset: 数据集实体
            children_map: 父子关系映射

        Returns:
            节点字典
        """
        # 获取统计信息
        statistics = await DatasetService._get_or_calculate_statistics(db, redis, dataset.id)

        # 先定义 children 列表，确保类型正确推断
        children_list: list[dict[str, Any]] = []

        node_dict = {
            "id": dataset.id,
            "parentId": dataset.parent_id,
            "treePath": dataset.tree_path,
            "type": dataset.type,
            "name": dataset.name,
            "img": dataset.img,
            "description": dataset.description,
            "path": dataset.path,
            "size": dataset.size,
            "status": dataset.status,
            "deleted": dataset.deleted,
            "createTime": format_time(dataset.create_time),
            "updateTime": format_time(dataset.update_time),
            "statistics": statistics,
            "children": children_list,
        }

        # 递归处理子节点
        children = children_map.get(dataset.id, [])
        for child in children:
            child_dict = await DatasetService._build_dataset_node(db, redis, child, children_map)
            children_list.append(child_dict)

        return node_dict

    @staticmethod
    async def _get_or_calculate_statistics(
        db: AsyncSession,
        redis: Redis,
        dataset_id: int,
    ) -> dict[str, Any] | None:
        """
        获取或计算数据集统计信息（带缓存）

        Args:
            db: 数据库会话
            redis: Redis 客户端
            dataset_id: 数据集ID

        Returns:
            统计信息字典
        """
        if dataset_id == 0:
            return None

        cache_key = f"{DatasetService.CACHE_STATS_PREFIX}{dataset_id}"

        # 尝试从缓存获取
        try:
            cached_data = await redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception:
            pass

        # 计算统计信息
        leaf_ids = await dataset_repository.get_leaf_ids(db, dataset_id)
        stats = await dataset_repository.calculate_statistics(db, leaf_ids)

        # 缓存结果
        try:
            await redis.setex(cache_key, DatasetService.CACHE_STATS_TTL, json.dumps(stats, ensure_ascii=False))
        except Exception:
            pass

        return stats

    @staticmethod
    async def get_dataset_options(db: AsyncSession) -> list[dict[str, Any]]:
        """
        获取数据集下拉选项（树形结构）

        Args:
            db: 数据库会话

        Returns:
            下拉选项列表
        """
        return await dataset_repository.get_dataset_options(db)

    @staticmethod
    async def get_dataset_by_id(
        db: AsyncSession,
        redis: Redis,
        dataset_id: int,
    ) -> dict[str, Any] | None:
        """
        根据ID获取数据集详情

        Args:
            db: 数据库会话
            redis: Redis 客户端
            dataset_id: 数据集ID

        Returns:
            数据集详情字典
        """
        dataset = await dataset_repository.get_by_id(db, dataset_id)

        if not dataset:
            return None

        statistics = await DatasetService._get_or_calculate_statistics(db, redis, dataset_id)

        return {
            "id": dataset.id,
            "parentId": dataset.parent_id,
            "treePath": dataset.tree_path,
            "type": dataset.type,
            "name": dataset.name,
            "img": dataset.img,
            "description": dataset.description,
            "path": dataset.path,
            "size": dataset.size,
            "status": dataset.status,
            "deleted": dataset.deleted,
            "createTime": format_time(dataset.create_time),
            "updateTime": format_time(dataset.update_time),
            "statistics": statistics,
        }

    @staticmethod
    async def create_dataset(
        db: AsyncSession,
        redis: Redis,
        data: dict[str, Any],
    ) -> int:
        """
        创建数据集

        Args:
            db: 数据库会话
            redis: Redis 客户端
            data: 数据集数据

        Returns:
            创建的数据集ID

        Raises:
            BusinessException: 父数据集不存在
        """
        parent_id = data.get("parentId", 0)
        name = data.get("name", "")

        # 验证父数据集是否存在
        if parent_id != 0:
            parent = await dataset_repository.get_by_id(db, parent_id)
            if not parent:
                raise BusinessException(
                    ResultCode.RESOURCE_NOT_FOUND, "父数据集不存在")

        # 名称唯一性校验：同一父节点下名称不重复
        if name:
            exists = await dataset_repository.check_name_exists(db, parent_id, name)
            if exists:
                raise BusinessException(
                    ResultCode.PARAM_ERROR, "同一层级下数据集名称已存在")

        # 生成树路径
        tree_path = await DatasetService._generate_tree_path(db, parent_id)

        # 创建数据集
        dataset = SysDataset(
            parent_id=parent_id,
            tree_path=tree_path,
            type=data.get("type", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            path=data.get("path", ""),
            status=data.get("status", 1),
            deleted=0,
        )

        db.add(dataset)
        await db.flush()
        await db.refresh(dataset)

        # 清除缓存
        await DatasetService._evict_tree_cache(redis)
        if parent_id != 0:
            await DatasetService._evict_dataset_and_ancestor_stats_cache(db, redis, parent_id)

        return dataset.id

    @staticmethod
    async def _generate_tree_path(db: AsyncSession, parent_id: int) -> str:
        """生成树路径"""
        if parent_id == 0:
            return "0"
        tree_path = await dataset_repository.get_dataset_tree_path(db, parent_id)
        return generate_tree_path(tree_path, parent_id)

    @staticmethod
    async def update_dataset(
        db: AsyncSession,
        redis: Redis,
        dataset_id: int,
        data: dict[str, Any],
    ) -> int:
        """
        更新数据集

        Args:
            db: 数据库会话
            redis: Redis 客户端
            dataset_id: 数据集ID
            data: 更新数据

        Returns:
            更新的数据集ID

        Raises:
            BusinessException: 数据集不存在、新父数据集不存在或循环引用
        """
        dataset = await dataset_repository.get_by_id(db, dataset_id)

        if not dataset:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据集不存在")

        old_parent_id = dataset.parent_id
        new_parent_id = data.get("parentId")

        # 处理父节点变更
        if new_parent_id is not None and new_parent_id != old_parent_id:
            # 验证新父数据集
            if new_parent_id != 0:
                new_parent = await dataset_repository.get_by_id(db, new_parent_id)
                if not new_parent:
                    raise BusinessException(
                        ResultCode.RESOURCE_NOT_FOUND, "新父数据集不存在")

            # 防止循环引用
            if await DatasetService._would_create_cycle(db, dataset_id, new_parent_id):
                raise BusinessException(
                    ResultCode.PARAM_ERROR, "不能将数据集移动到其子节点下")

            # 更新树路径
            old_tree_path = dataset.tree_path
            new_tree_path = await DatasetService._generate_tree_path(db, new_parent_id)
            dataset.tree_path = new_tree_path
            dataset.parent_id = new_parent_id

            # 更新所有子节点的树路径
            await DatasetService._update_children_tree_paths(db, dataset_id, old_tree_path, new_tree_path)

            # 清除缓存
            await DatasetService._evict_dataset_stats_cache(redis, dataset_id)
            if old_parent_id != 0:
                await DatasetService._evict_dataset_and_ancestor_stats_cache(db, redis, old_parent_id)
            if new_parent_id != 0:
                await DatasetService._evict_dataset_and_ancestor_stats_cache(db, redis, new_parent_id)

        # 名称唯一性校验
        if "name" in data and data["name"] != dataset.name:
            check_parent = new_parent_id if new_parent_id is not None else old_parent_id
            exists = await dataset_repository.check_name_exists(
                db, check_parent, data["name"], exclude_id=dataset_id,
            )
            if exists:
                raise BusinessException(
                    ResultCode.PARAM_ERROR, "同一层级下数据集名称已存在")

        # 更新其他字段
        if "name" in data:
            dataset.name = data["name"]
        if "type" in data:
            dataset.type = data["type"]
        if "description" in data:
            dataset.description = data["description"]
        if "path" in data:
            dataset.path = data["path"]
        if "status" in data:
            dataset.status = data["status"]

        dataset.update_time = datetime.now()

        await DatasetService._evict_tree_cache(redis)

        return dataset_id

    @staticmethod
    async def _would_create_cycle(db: AsyncSession, dataset_id: int, new_parent_id: int) -> bool:
        """
        检查是否会产生循环引用

        Args:
            db: 数据库会话
            dataset_id: 当前数据集ID
            new_parent_id: 新父节点ID

        Returns:
            是否会产生循环引用
        """
        if new_parent_id == 0:
            return False

        # 检查新父节点是否是当前节点的后代
        descendants = await dataset_repository.get_all_descendant_ids(db, dataset_id)
        return new_parent_id in descendants

    @staticmethod
    async def _update_children_tree_paths(
        db: AsyncSession,
        dataset_id: int,
        old_prefix: str,
        new_prefix: str,
    ):
        """
        更新所有子节点的树路径

        Args:
            db: 数据库会话
            dataset_id: 数据集ID
            old_prefix: 旧路径前缀
            new_prefix: 新路径前缀
        """
        children = await dataset_repository.get_all_descendant_ids(db, dataset_id)

        for child_id in children:
            child = await dataset_repository.get_by_id(db, child_id, with_deleted=True)
            if child and child.tree_path and child.tree_path.startswith(old_prefix):
                suffix = child.tree_path[len(old_prefix):]
                child.tree_path = f"{new_prefix}{suffix}"

    @staticmethod
    async def delete_datasets(
        db: AsyncSession,
        redis: Redis,
        dataset_ids: list[int],
    ) -> dict[str, Any]:
        """
        批量删除数据集（级联删除子数据集、数据项、文件）

        Args:
            db: 数据库会话
            redis: Redis 客户端
            dataset_ids: 数据集ID列表

        Returns:
            删除结果
        """
        if not dataset_ids:
            raise BusinessException(ResultCode.PARAM_ERROR, "未指定要删除的数据集")

        total = len(dataset_ids)
        succeeded = 0
        failed = 0
        results = []

        for dataset_id in dataset_ids:
            try:
                # 获取数据集
                dataset = await dataset_repository.get_by_id(db, dataset_id, with_deleted=True)

                if not dataset:
                    failed += 1
                    results.append({
                        "datasetId": dataset_id,
                        "status": "failed",
                        "message": "数据集不存在",
                    })
                    continue

                parent_id = dataset.parent_id

                # 获取所有需要删除的数据集ID（包括子节点）
                all_dataset_ids = await DatasetService._get_dataset_and_descendant_ids(db, dataset_id)

                # 找出叶子节点
                all_datasets = await dataset_repository.get_all_datasets_for_tree_path_update(db, all_dataset_ids)

                children_map: dict[int, list[SysDataset]] = {}
                for ds in all_datasets:
                    if ds.parent_id not in children_map:
                        children_map[ds.parent_id] = []
                    children_map[ds.parent_id].append(ds)

                leaf_ids = [
                    ds_id for ds_id in all_dataset_ids if not children_map.get(ds_id)]

                # 删除所有数据项（包括关联的文件记录）
                for leaf_id in leaf_ids:
                    await DatasetItemService.delete_items_by_dataset(db, redis, leaf_id)

                # 从叶子节点往上删除数据集（按深度排序）
                depth_map = await dataset_repository.get_dataset_depth(db, all_dataset_ids)

                sorted_ids = sorted(
                    all_dataset_ids, key=lambda x: depth_map.get(x, 0), reverse=True)

                # 批量删除数据集
                await dataset_repository.delete_by_ids(db, sorted_ids)

                succeeded += 1

                # 清除缓存
                for deleted_id in all_dataset_ids:
                    await DatasetService._evict_dataset_stats_cache(redis, deleted_id)
                if parent_id and parent_id != 0:
                    await DatasetService._evict_dataset_and_ancestor_stats_cache(db, redis, parent_id)

                results.append({
                    "datasetId": dataset_id,
                    "status": "success",
                })

            except Exception as e:
                failed += 1
                results.append({
                    "datasetId": dataset_id,
                    "status": "failed",
                    "message": str(e),
                })

        await DatasetService._evict_tree_cache(redis)

        return {
            "success": True,
            "message": f"删除完成：成功 {succeeded} 个，失败 {failed} 个",
            "data": {
                "total": total,
                "succeeded": succeeded,
                "failed": failed,
                "results": results,
            },
        }

    @staticmethod
    async def _get_dataset_and_descendant_ids(db: AsyncSession, dataset_id: int) -> list[int]:
        """
        获取数据集及其所有后代ID

        Args:
            db: 数据库会话
            dataset_id: 数据集ID

        Returns:
            数据集ID列表（包括自己和所有子节点）
        """
        descendants = await dataset_repository.get_all_descendant_ids(db, dataset_id)
        return [dataset_id] + descendants

    @staticmethod
    async def _evict_dataset_stats_cache(redis: Redis, dataset_id: int):
        """
        清除指定数据集的统计缓存

        Args:
            redis: Redis 客户端
            dataset_id: 数据集ID
        """
        cache_key = f"{DatasetService.CACHE_STATS_PREFIX}{dataset_id}"
        try:
            await redis.delete(cache_key)
        except Exception:
            pass

    @staticmethod
    async def _evict_dataset_and_ancestor_stats_cache(
        db: AsyncSession,
        redis: Redis,
        dataset_id: int,
    ):
        """
        清除数据集及其祖先的统计缓存

        Args:
            db: 数据库会话
            redis: Redis 客户端
            dataset_id: 数据集ID
        """
        tree_path = await dataset_repository.get_dataset_tree_path(db, dataset_id)

        if not tree_path:
            return

        # tree_path 格式为 "0,1,2,3"，需要反转为 [3, 2, 1, 0]
        ancestor_ids = [int(x) for x in reversed(tree_path.split(","))]

        for ancestor_id in ancestor_ids:
            await DatasetService._evict_dataset_stats_cache(redis, ancestor_id)

    @staticmethod
    async def _evict_tree_cache(redis: Redis):
        """清除数据集树缓存"""
        try:
            await redis.delete(DatasetService.CACHE_TREE_KEY)
        except Exception:
            pass

    @staticmethod
    async def get_image_items(
        db: AsyncSession,
        redis: Redis,
        dataset_id: int,
        page_num: int = 1,
        page_size: int = 20,
        keywords: str | None = None,
    ) -> dict[str, Any]:
        """
        获取数据集下的数据项（分页）

        Args:
            db: 数据库会话
            redis: Redis 客户端
            dataset_id: 数据集ID
            page_num: 页码
            page_size: 每页数量
            keywords: 搜索关键词

        Returns:
            分页结果
        """
        # 获取叶子节点
        leaf_ids = await dataset_repository.get_leaf_ids(db, dataset_id)

        # 查询总数
        total = await dataset_repository.get_items_count(db, leaf_ids, keywords)

        # 分页查询
        offset = (page_num - 1) * page_size
        items = await dataset_repository.get_items_paginated(
            db, leaf_ids, offset, page_size, keywords,
        )

        # 构建返回数据
        records = []
        for item in items:
            item_vo = await DatasetItemService.get_item_detail(db, item.id)
            if item_vo:
                records.append(item_vo)

        return {
            "list": records,
            "total": total,
            "pageNum": page_num,
            "pageSize": page_size,
        }


class DatasetItemService:
    """数据集项服务（异步版本）"""

    @staticmethod
    async def create_dataset_item(
        db: AsyncSession,
        redis: Redis,
        data: dict[str, Any],
    ) -> int:
        """
        创建数据集项

        Args:
            db: 数据库会话
            redis: Redis 客户端
            data: 数据项数据

        Returns:
            创建的数据项ID

        Raises:
            BusinessException: 数据集不存在或为目录类型
        """
        dataset_id = data.get("datasetId")

        if not dataset_id:
            raise BusinessException(ResultCode.PARAM_ERROR, "数据集ID不能为空")

        # 验证数据集存在且不是目录
        dataset = await dataset_repository.get_by_id(db, dataset_id)

        if not dataset:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据集不存在")

        # 检查是否是叶子节点
        children_count = await dataset_repository.get_children_count(db, dataset_id)

        if children_count > 0:
            raise BusinessException(
                ResultCode.PARAM_ERROR, "不能在目录类型的数据集中创建数据项")

        # 创建数据项
        dataset_item = SysDatasetItem(
            dataset_id=dataset_id,
            name=data.get("name", ""),
        )

        db.add(dataset_item)
        await db.flush()
        await db.refresh(dataset_item)

        # 清除数据集统计缓存
        await DatasetService._evict_dataset_stats_cache(redis, dataset_id)

        return dataset_item.id

    @staticmethod
    async def get_item_detail(db: AsyncSession, item_id: int) -> dict[str, Any]:
        """
        获取数据项详情

        Args:
            db: 数据库会话
            item_id: 数据项ID

        Returns:
            数据项详情
        """
        item, item_files = await dataset_repository.get_item_with_files(db, item_id)

        if not item:
            return {}

        files = []
        image_urls = []

        for item_file, file_obj in item_files:
            files.append({
                "id": item_file.id,
                "itemId": item_file.item_id,
                "fileId": item_file.file_id,
                "type": item_file.type,
                "sceneType": item_file.scene_type,
                "hazeLevel": item_file.haze_level,
                "description": item_file.description,
                "url": file_obj.url,
                "name": file_obj.name,
                "size": file_obj.size,
                "md5": file_obj.md5,
            })
            image_urls.append({
                "id": file_obj.id,
                "type": item_file.type,
                "url": file_obj.url,
                "thumbnailUrl": file_obj.url,
            })

        return {
            "id": item.id,
            "datasetId": item.dataset_id,
            "name": item.name,
            "createTime": format_time(item.create_time) if hasattr(item, "create_time") else None,
            "updateTime": format_time(item.update_time) if hasattr(item, "update_time") else None,
            "files": files,
            "imgUrl": image_urls,
        }

    @staticmethod
    async def update_dataset_item(
        db: AsyncSession,
        redis: Redis,
        item_id: int,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        更新数据集项

        Args:
            db: 数据库会话
            redis: Redis 客户端
            item_id: 数据项ID
            data: 更新数据

        Returns:
            更新结果
        """
        item = await dataset_repository.get_item_by_id(db, item_id)

        if not item:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据项不存在")

        if "name" in data:
            item.name = data["name"]

        item.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 清除数据集统计缓存
        await DatasetService._evict_dataset_stats_cache(redis, item.dataset_id)

        return {"id": item_id}

    @staticmethod
    async def delete_dataset_item(
        db: AsyncSession,
        redis: Redis,
        item_id: int,
    ):
        """
        删除数据项

        Args:
            db: 数据库会话
            redis: Redis 客户端
            item_id: 数据项ID

        Returns:
            删除结果
        """
        item = await dataset_repository.get_item_by_id(db, item_id)

        if not item:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据项不存在")

        dataset_id = item.dataset_id

        # 删除关联的文件项记录
        await dataset_repository.delete_item_files_by_item_id(db, item_id)

        # 删除数据项
        await dataset_repository.delete_item_by_id(db, item_id)

        # 清除数据集统计缓存
        await DatasetService._evict_dataset_stats_cache(redis, dataset_id)

    @staticmethod
    async def delete_items_by_dataset(
        db: AsyncSession,
        redis: Redis,
        dataset_id: int,
    ) -> int:
        """
        删除数据集下的所有数据项

        Args:
            db: 数据库会话
            redis: Redis 客户端
            dataset_id: 数据集ID

        Returns:
            操作结果
        """
        # 获取所有数据项ID
        item_ids = await dataset_repository.get_item_ids_by_dataset(db, dataset_id)

        if not item_ids:
            return 0

        # 删除关联的文件项记录
        await dataset_repository.delete_item_files_by_item_ids(db, item_ids)

        # 删除数据项
        await dataset_repository.delete_items_by_dataset_id(db, dataset_id)

        return len(item_ids)

    @staticmethod
    async def batch_delete_items(
        db: AsyncSession,
        redis: Redis,
        item_ids: list[int],
    ):
        """批量删除数据项"""
        if not item_ids:
            raise BusinessException(ResultCode.PARAM_ERROR, "未指定要删除的数据项")

        affected_dataset_ids: set[int] = set()

        for item_id in item_ids:
            item = await dataset_repository.get_item_by_id(db, item_id)
            if not item:
                continue
            affected_dataset_ids.add(item.dataset_id)
            await dataset_repository.delete_item_files_by_item_id(db, item_id)
            await dataset_repository.delete_item_by_id(db, item_id)

        # 清除受影响数据集的统计缓存
        for ds_id in affected_dataset_ids:
            await DatasetService._evict_dataset_stats_cache(redis, ds_id)

    @staticmethod
    async def upload_dataset_item_with_images(
        db: AsyncSession,
        redis: Redis,
        dataset_id: int,
        name: str | None = None,
        scene_type: str | None = None,
        clear_file_content: bytes | None = None,
        clear_filename: str = "",
        clear_content_type: str = "",
        hazy_files_data: list[dict] | None = None,
    ) -> dict:
        """创建数据项并上传配对图片（一张清晰图 + 多张有雾图）"""
        if clear_file_content is None:
            raise BusinessException(ResultCode.PARAM_ERROR, "清晰图必须上传")
        if not hazy_files_data:
            raise BusinessException(ResultCode.PARAM_ERROR, "至少上传一张有雾图")

        # 校验数据集存在
        dataset = await dataset_repository.get_by_id(db, dataset_id)
        if not dataset or dataset.deleted:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据集不存在")
        if dataset.type == "DIR":
            raise BusinessException(ResultCode.PARAM_ERROR, "目录类型数据集不允许创建数据项")

        # 创建数据项
        item_name = name or f"Item_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        item = SysDatasetItem(dataset_id=dataset_id, name=item_name)
        db.add(item)
        await db.flush()
        await db.refresh(item)

        # 上传清晰图
        clear_sys_file = await FileService.upload_file(
            db, clear_filename, clear_file_content, clear_content_type,
        )
        item_file_clear = SysItemFile(
            item_id=item.id,
            file_id=clear_sys_file.id,
            type="clear",
            scene_type=scene_type or "",
            haze_level="",
        )
        db.add(item_file_clear)

        # 上传有雾图
        for hfd in hazy_files_data:
            haze_level = hfd.get("hazeLevel", "medium").lower()
            if haze_level not in ("light", "medium", "heavy"):
                haze_level = "medium"

            hazy_sys_file = await FileService.upload_file(
                db, hfd["filename"], hfd["content"], hfd.get("contentType", "application/octet-stream"),
            )
            item_file_hazy = SysItemFile(
                item_id=item.id,
                file_id=hazy_sys_file.id,
                type="hazy",
                scene_type=scene_type or "",
                haze_level=haze_level,
            )
            db.add(item_file_hazy)

        await db.flush()

        # 清除缓存
        await DatasetService._evict_dataset_stats_cache(redis, dataset_id)
        await DatasetService._evict_tree_cache(redis)

        # 返回详情
        return await DatasetItemService.get_item_detail(db, item.id)

    @staticmethod
    async def batch_create_dataset_items_with_images(
        db: AsyncSession,
        redis: Redis,
        dataset_id: int,
        scene_type: str | None = None,
        files_data: list[dict] | None = None,
    ) -> dict:
        """批量创建数据项并上传图片（按文件名自动配对）"""
        if not files_data:
            raise BusinessException(ResultCode.PARAM_ERROR, "至少上传一个文件")

        # 校验数据集存在
        dataset = await dataset_repository.get_by_id(db, dataset_id)
        if not dataset or dataset.deleted:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据集不存在")
        if dataset.type == "DIR":
            raise BusinessException(ResultCode.PARAM_ERROR, "目录类型数据集不允许创建数据项")

        # 按文件名前缀分组
        groups: dict[str, dict[str, list]] = {}
        unpaired: list[dict] = []

        for fd in files_data:
            filename = fd["filename"]
            clear = _is_clear_image(filename)
            hazy = _is_hazy_image(filename)

            if not clear and not hazy:
                unpaired.append({"fileName": filename, "reason": "无法识别文件类型，文件名需包含 _clear/_gt 或 _hazy"})
                continue

            if clear and hazy:
                unpaired.append({"fileName": filename, "reason": "文件名同时包含清晰图和有雾图标识，无法判定"})
                continue

            prefix = _extract_file_prefix(filename)
            if not prefix:
                unpaired.append({"fileName": filename, "reason": "无法提取文件名前缀"})
                continue

            if prefix not in groups:
                groups[prefix] = {"clear": [], "hazy": []}

            if clear:
                groups[prefix]["clear"].append(fd)
            else:
                haze_level = _extract_haze_level(filename)
                fd["hazeLevel"] = haze_level
                groups[prefix]["hazy"].append(fd)

        # 处理每个配对组
        success_items = []
        failed_items = []
        total = len(groups)

        for prefix, files in groups.items():
            if not files["clear"]:
                failed_items.append({
                    "fileName": prefix,
                    "reason": f"未找到清晰图（需要 {prefix}_clear 或 {prefix}_gt 文件）",
                })
                continue
            if not files["hazy"]:
                failed_items.append({
                    "fileName": prefix,
                    "reason": f"未找到有雾图（需要 {prefix}_hazy 文件）",
                })
                continue

            try:
                clear_fd = files["clear"][0]
                details = await DatasetItemService.upload_dataset_item_with_images(
                    db=db,
                    redis=redis,
                    dataset_id=dataset_id,
                    name=prefix,
                    scene_type=scene_type,
                    clear_file_content=clear_fd["content"],
                    clear_filename=clear_fd["filename"],
                    clear_content_type=clear_fd.get("contentType", "application/octet-stream"),
                    hazy_files_data=files["hazy"],
                )
                file_count = len(details.get("files", [])) if details else 0
                success_items.append({
                    "id": details["id"] if details else 0,
                    "name": details.get("name"),
                    "fileCount": file_count,
                })
            except Exception as e:
                failed_items.append({
                    "fileName": prefix,
                    "reason": str(e),
                })

        # 添加未配对文件到失败列表
        failed_items.extend(unpaired)

        succeeded = len(success_items)
        failed = len(failed_items)

        return {
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "successItems": success_items,
            "failedItems": failed_items,
        }


class ItemFileService:
    """图片文件服务"""

    @staticmethod
    async def get_item_file_detail(
        db: AsyncSession,
        file_id: int,
    ) -> dict[str, Any] | None:
        """获取图片文件详情"""
        result = await dataset_repository.get_item_file_with_file(db, file_id)
        if not result:
            return None

        item_file, file_obj = result
        return {
            "id": item_file.id,
            "itemId": item_file.item_id,
            "fileId": item_file.file_id,
            "type": item_file.type,
            "sceneType": item_file.scene_type,
            "hazeLevel": item_file.haze_level,
            "description": item_file.description,
            "url": file_obj.url if file_obj else None,
            "thumbnailUrl": file_obj.url if file_obj else None,
            "name": file_obj.name if file_obj else None,
            "size": file_obj.size if file_obj else None,
            "md5": file_obj.md5 if file_obj else None,
        }

    @staticmethod
    async def upload_item_file(
        db: AsyncSession,
        redis: Redis,
        item_id: int,
        image_type: str,
        scene_type: str,
        haze_level: str,
        description: str,
        file,
    ) -> dict[str, Any]:
        """上传数据项图片（含文件上传 + 关联记录创建）"""
        from app.service.file_service import FileService

        # 校验数据项存在
        item = await dataset_repository.get_item_by_id(db, item_id)
        if not item:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据项不存在")

        # 校验图片类型
        valid_types = {"clear", "hazy", "depth", "segment"}
        if image_type not in valid_types:
            raise BusinessException(
                ResultCode.PARAM_ERROR, f"不支持的图片类型: {image_type}")

        # 校验雾霾等级
        if image_type == "hazy" and haze_level:
            valid_levels = {"light", "medium", "heavy"}
            if haze_level not in valid_levels:
                raise BusinessException(
                    ResultCode.PARAM_ERROR, f"不支持的雾霾等级: {haze_level}")

        # 读取文件内容并上传
        content = await file.read()
        if not file.filename:
            raise BusinessException(ResultCode.PARAM_ERROR, "文件名不能为空")

        file_info = await FileService.upload_file(
            db=db,
            filename=file.filename,
            content=content,
            content_type=file.content_type or "application/octet-stream",
        )

        # 创建关联记录
        item_file = SysItemFile(
            item_id=item_id,
            file_id=file_info.id,
            type=image_type,
            scene_type=scene_type or "未分类",
            haze_level=haze_level or "未标注",
            description=description,
        )
        db.add(item_file)
        await db.flush()
        await db.refresh(item_file)

        # 清除统计缓存
        await DatasetService._evict_dataset_stats_cache(redis, item.dataset_id)

        return {
            "id": item_file.id,
            "itemId": item_file.item_id,
            "fileId": item_file.file_id,
            "type": item_file.type,
            "sceneType": item_file.scene_type,
            "hazeLevel": item_file.haze_level,
            "description": item_file.description,
            "url": file_info.url,
            "name": file_info.name,
            "size": file_info.size_bytes,
            "md5": file_info.md5,
        }

    @staticmethod
    async def update_item_file(
        db: AsyncSession,
        redis: Redis,
        file_id: int,
        data: dict[str, Any],
    ):
        """修改图片元数据"""
        item_file = await dataset_repository.get_item_file_by_id(db, file_id)
        if not item_file:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "图片文件不存在")

        if "type" in data:
            item_file.type = data["type"]
        if "sceneType" in data:
            item_file.scene_type = data["sceneType"]
        if "hazeLevel" in data:
            item_file.haze_level = data["hazeLevel"]
        if "description" in data:
            item_file.description = data["description"]

        # 获取数据项以清除对应数据集缓存
        item = await dataset_repository.get_item_by_id(db, item_file.item_id)
        if item:
            await DatasetService._evict_dataset_stats_cache(redis, item.dataset_id)

    @staticmethod
    async def delete_item_file(
        db: AsyncSession,
        redis: Redis,
        file_id: int,
    ):
        """删除单个图片文件关联"""
        item_file = await dataset_repository.get_item_file_by_id(db, file_id)
        if not item_file:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "图片文件不存在")

        dataset_id = None
        item = await dataset_repository.get_item_by_id(db, item_file.item_id)
        if item:
            dataset_id = item.dataset_id

        await dataset_repository.delete_item_file_by_id(db, file_id)

        if dataset_id:
            await DatasetService._evict_dataset_stats_cache(redis, dataset_id)

    @staticmethod
    async def batch_delete_item_files(
        db: AsyncSession,
        redis: Redis,
        file_ids: list[int],
    ):
        """批量删除图片文件关联"""
        if not file_ids:
            raise BusinessException(ResultCode.PARAM_ERROR, "未指定要删除的图片")

        affected_dataset_ids: set[int] = set()

        for fid in file_ids:
            item_file = await dataset_repository.get_item_file_by_id(db, fid)
            if not item_file:
                continue
            item = await dataset_repository.get_item_by_id(db, item_file.item_id)
            if item:
                affected_dataset_ids.add(item.dataset_id)

        await dataset_repository.delete_item_files_by_ids(db, file_ids)

        for ds_id in affected_dataset_ids:
            await DatasetService._evict_dataset_stats_cache(redis, ds_id)
            await DatasetService._evict_dataset_stats_cache(redis, ds_id)
