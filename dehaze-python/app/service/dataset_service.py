"""
数据集服务

提供数据集 CRUD 功能，支持树形结构、数据项管理
性能优化：
- 叶子节点批量查询统计，避免N+1问题
- 内存向上聚合统计数据
- 全局缓存（所有数据集、所有统计信息）
- 分页查询+懒加载子节点
- 数据项批量查询文件信息，避免N+1
"""

import json
import logging
import re
import time
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


def _create_empty_stats() -> dict[str, Any]:
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


def _merge_stats(parent: dict[str, Any], child: dict[str, Any]):
    parent["itemCount"] += child.get("itemCount", 0)
    parent["fileCount"] += child.get("fileCount", 0)
    parent["totalSize"] += child.get("totalSize", 0)
    parent["clearCount"] += child.get("clearCount", 0)
    parent["hazyCount"] += child.get("hazyCount", 0)
    for k, v in child.get("sceneDistribution", {}).items():
        parent["sceneDistribution"][k] = parent["sceneDistribution"].get(k, 0) + v
    for k, v in child.get("hazeDistribution", {}).items():
        parent["hazeDistribution"][k] = parent["hazeDistribution"].get(k, 0) + v
    for k, v in child.get("formatDistribution", {}).items():
        parent["formatDistribution"][k] = parent["formatDistribution"].get(k, 0) + v


class DatasetService:
    """数据集服务（异步版本，性能优化版）"""

    CACHE_ALL_KEY = "dataset:all"
    CACHE_STATSMAP_KEY = "dataset:statsMap:all"
    CACHE_TREE_KEY = "dataset:tree"
    CACHE_OPTIONS_KEY = "dataset:tree:options"
    CACHE_ALL_TTL = 3600
    CACHE_STATS_TTL = 1800
    CACHE_TREE_TTL = 3600

    ROOT_NODE_ID = 0

    @staticmethod
    async def _evict_all_cache(redis: Redis):
        keys = [
            DatasetService.CACHE_ALL_KEY,
            DatasetService.CACHE_STATSMAP_KEY,
            DatasetService.CACHE_TREE_KEY,
            DatasetService.CACHE_OPTIONS_KEY,
        ]
        for key in keys:
            try:
                await redis.delete(key)
            except Exception as e:
                logger.warning(f"清除缓存失败 {key}: {e}")

    @staticmethod
    async def get_all_datasets(db: AsyncSession, redis: Redis) -> list[SysDataset]:
        try:
            cached = await redis.get(DatasetService.CACHE_ALL_KEY)
            if cached:
                data = json.loads(cached)
                result = []
                for item in data:
                    ds = SysDataset()
                    for k, v in item.items():
                        if hasattr(ds, k):
                            setattr(ds, k, v)
                    result.append(ds)
                if result:
                    return result
        except Exception:
            pass

        datasets = await dataset_repository.find_all(db)

        try:
            serializable = []
            for ds in datasets:
                serializable.append({
                    "id": ds.id,
                    "parent_id": ds.parent_id,
                    "tree_path": ds.tree_path,
                    "type": ds.type,
                    "name": ds.name,
                    "img": ds.img,
                    "description": ds.description,
                    "path": ds.path,
                    "size": ds.size,
                    "status": ds.status,
                    "deleted": ds.deleted,
                })
            await redis.setex(
                DatasetService.CACHE_ALL_KEY,
                DatasetService.CACHE_ALL_TTL,
                json.dumps(serializable, ensure_ascii=False, default=str),
            )
        except Exception as e:
            logger.warning(f"缓存写入失败: {e}")

        return datasets

    @staticmethod
    async def get_all_dataset_stats(db: AsyncSession, redis: Redis) -> dict[int, dict[str, Any]]:
        try:
            cached = await redis.get(DatasetService.CACHE_STATSMAP_KEY)
            if cached:
                return json.loads(cached)
        except Exception:
            pass

        start_time = time.time()
        logger.debug("开始计算所有数据集统计信息...")

        all_datasets = await DatasetService.get_all_datasets(db, redis)

        stats_map: dict[int, dict[str, Any]] = {}
        for ds in all_datasets:
            stats_map[int(ds.id)] = _create_empty_stats()

        if not all_datasets:
            return stats_map

        parent_ids_set: set[int] = set()
        for d in all_datasets:
            pid = int(d.parent_id)
            if pid != 0:
                parent_ids_set.add(pid)

        leaf_ids: list[int] = []
        for d in all_datasets:
            did = int(d.id)
            if did not in parent_ids_set:
                leaf_ids.append(did)

        if leaf_ids:
            logger.debug(f"发现叶子数据集 {len(leaf_ids)} 个")

            item_counts = await dataset_repository.count_items_per_dataset(db, leaf_ids)
            for ds_id, cnt in item_counts.items():
                if ds_id in stats_map:
                    stats_map[ds_id]["itemCount"] = cnt

            stats_results = await dataset_repository.count_dataset_stats_batch(db, leaf_ids)
            for ds_id, st in stats_results.items():
                if ds_id in stats_map:
                    stats_map[ds_id]["fileCount"] = st["fileCount"]
                    stats_map[ds_id]["totalSize"] = st["totalSize"]
                    stats_map[ds_id]["clearCount"] = st["clearCount"]
                    stats_map[ds_id]["hazyCount"] = st["hazyCount"]

            scene_results = await dataset_repository.count_scene_distribution_batch(db, leaf_ids)
            for ds_id, dist in scene_results.items():
                if ds_id in stats_map:
                    stats_map[ds_id]["sceneDistribution"] = dist

            haze_results = await dataset_repository.count_haze_distribution_batch(db, leaf_ids)
            for ds_id, dist in haze_results.items():
                if ds_id in stats_map:
                    stats_map[ds_id]["hazeDistribution"] = dist

            format_results = await dataset_repository.count_format_distribution_batch(db, leaf_ids)
            for ds_id, dist in format_results.items():
                if ds_id in stats_map:
                    stats_map[ds_id]["formatDistribution"] = dist

        parent_to_children: dict[int, list[int]] = {}
        id_to_dataset: dict[int, SysDataset] = {}
        for ds in all_datasets:
            did = int(ds.id)
            pid = int(ds.parent_id)
            id_to_dataset[did] = ds
            if pid != 0:
                if pid not in parent_to_children:
                    parent_to_children[pid] = []
                parent_to_children[pid].append(did)

        processed: set[int] = set(leaf_ids)
        queue: list[int] = list(leaf_ids)

        while queue:
            current_id = queue.pop(0)
            current = id_to_dataset.get(current_id)
            if not current:
                continue
            parent_id = int(current.parent_id)
            if parent_id == 0:
                continue

            parent_stats = stats_map.get(parent_id)
            child_stats = stats_map.get(current_id)
            if parent_stats and child_stats:
                _merge_stats(parent_stats, child_stats)

            siblings = parent_to_children.get(parent_id, [])
            all_siblings_processed = all(sid in processed for sid in siblings)
            if all_siblings_processed and parent_id not in processed:
                processed.add(parent_id)
                queue.append(parent_id)

        cost_ms = int((time.time() - start_time) * 1000)
        logger.info(f"所有数据集统计信息计算完成，耗时 {cost_ms} ms，叶子节点 {len(leaf_ids)} 个")

        try:
            str_key_map = {str(k): v for k, v in stats_map.items()}
            await redis.setex(
                DatasetService.CACHE_STATSMAP_KEY,
                DatasetService.CACHE_STATS_TTL,
                json.dumps(str_key_map, ensure_ascii=False),
            )
        except Exception as e:
            logger.warning(f"统计缓存写入失败: {e}")

        return stats_map

    @staticmethod
    def _entity_to_vo(
        entity: SysDataset,
        stats: dict[str, Any] | None,
        has_children: bool,
    ) -> dict[str, Any]:
        vo: dict[str, Any] = {
            "id": entity.id,
            "parentId": entity.parent_id,
            "treePath": entity.tree_path,
            "type": entity.type,
            "name": entity.name,
            "img": entity.img,
            "description": entity.description,
            "path": entity.path,
            "size": entity.size,
            "hasChildren": has_children,
            "children": [],
            "status": entity.status,
            "statistics": stats,
            "createTime": format_time(entity.create_time),
            "updateTime": format_time(entity.update_time),
        }
        if stats:
            vo["total"] = stats.get("fileCount", 0)
        return vo

    @staticmethod
    async def get_page(
        db: AsyncSession,
        redis: Redis,
        page_num: int = 1,
        page_size: int = 10,
        keywords: str | None = None,
    ) -> dict[str, Any]:
        root_datasets, total = await dataset_repository.find_root_page(db, page_num, page_size, keywords)
        if not root_datasets:
            return {"list": [], "total": total, "pageNum": page_num, "pageSize": page_size}

        root_ids = [int(d.id) for d in root_datasets]

        direct_children = await dataset_repository.find_by_parent_ids(db, root_ids)
        direct_children_map: dict[int, list[SysDataset]] = {}
        child_ids: list[int] = []
        for c in direct_children:
            pid = int(c.parent_id)
            if pid not in direct_children_map:
                direct_children_map[pid] = []
            direct_children_map[pid].append(c)
            child_ids.append(int(c.id))

        all_parent_ids = root_ids + child_ids
        has_children_map = await dataset_repository.count_has_children(db, all_parent_ids)

        stats_map = await DatasetService.get_all_dataset_stats(db, redis)

        vo_list = []
        for root in root_datasets:
            root_id = int(root.id)
            root_stats = stats_map.get(root_id, _create_empty_stats())
            root_vo = DatasetService._entity_to_vo(root, root_stats, has_children_map.get(root_id, False))

            children = direct_children_map.get(root_id, [])
            child_vos = []
            for child in children:
                cid = int(child.id)
                c_stats = stats_map.get(cid, _create_empty_stats())
                child_vos.append(DatasetService._entity_to_vo(child, c_stats, has_children_map.get(cid, False)))
            root_vo["children"] = child_vos
            vo_list.append(root_vo)

        return {
            "list": vo_list,
            "total": total,
            "pageNum": page_num,
            "pageSize": page_size,
        }

    @staticmethod
    async def get_children(
        db: AsyncSession,
        redis: Redis,
        parent_id: int,
    ) -> list[dict[str, Any]]:
        if parent_id <= 0:
            return []

        children = await dataset_repository.find_by_parent_id(db, parent_id)
        if not children:
            return []

        child_ids = [int(c.id) for c in children]
        has_children_map = await dataset_repository.count_has_children(db, child_ids)
        stats_map = await DatasetService.get_all_dataset_stats(db, redis)

        result = []
        for child in children:
            cid = int(child.id)
            c_stats = stats_map.get(cid, _create_empty_stats())
            child_vo = DatasetService._entity_to_vo(child, c_stats, has_children_map.get(cid, False))
            child_vo["children"] = []
            result.append(child_vo)

        return result

    @staticmethod
    async def get_dataset_options(db: AsyncSession, redis: Redis) -> list[dict[str, Any]]:
        try:
            cached = await redis.get(DatasetService.CACHE_OPTIONS_KEY)
            if cached:
                return json.loads(cached)
        except Exception:
            pass

        options = await dataset_repository.get_dataset_options(db)

        try:
            await redis.setex(
                DatasetService.CACHE_OPTIONS_KEY,
                DatasetService.CACHE_TREE_TTL,
                json.dumps(options, ensure_ascii=False),
            )
        except Exception as e:
            logger.warning(f"选项缓存写入失败: {e}")

        return options

    @staticmethod
    async def get_dataset_by_id(
        db: AsyncSession,
        redis: Redis,
        dataset_id: int,
    ) -> dict[str, Any] | None:
        dataset = await dataset_repository.get_by_id(db, dataset_id)
        if not dataset:
            return None

        stats_map = await DatasetService.get_all_dataset_stats(db, redis)
        statistics = stats_map.get(int(dataset_id), _create_empty_stats())

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
        parent_id = data.get("parentId", 0)
        name = data.get("name", "")

        if parent_id != 0:
            parent = await dataset_repository.get_by_id(db, parent_id)
            if not parent:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "父数据集不存在")

        if name:
            exists = await dataset_repository.check_name_exists(db, parent_id, name)
            if exists:
                raise BusinessException(ResultCode.PARAM_ERROR, "同一层级下数据集名称已存在")

        tree_path = await DatasetService._generate_tree_path(db, parent_id)

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

        await DatasetService._evict_all_cache(redis)

        return dataset.id

    @staticmethod
    async def _generate_tree_path(db: AsyncSession, parent_id: int) -> str:
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
        dataset = await dataset_repository.get_by_id(db, dataset_id)
        if not dataset:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据集不存在")

        old_parent_id = dataset.parent_id
        new_parent_id = data.get("parentId")

        if new_parent_id is not None and new_parent_id != old_parent_id:
            if new_parent_id != 0:
                new_parent = await dataset_repository.get_by_id(db, new_parent_id)
                if not new_parent:
                    raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "新父数据集不存在")

            if await DatasetService._would_create_cycle(db, dataset_id, new_parent_id):
                raise BusinessException(ResultCode.PARAM_ERROR, "不能将数据集移动到其子节点下")

            old_tree_path = dataset.tree_path
            new_tree_path = await DatasetService._generate_tree_path(db, new_parent_id)
            dataset.tree_path = new_tree_path
            dataset.parent_id = new_parent_id
            await DatasetService._update_children_tree_paths(db, dataset_id, old_tree_path, new_tree_path)

        if "name" in data and data["name"] != dataset.name:
            check_parent = new_parent_id if new_parent_id is not None else old_parent_id
            exists = await dataset_repository.check_name_exists(
                db, check_parent, data["name"], exclude_id=dataset_id,
            )
            if exists:
                raise BusinessException(ResultCode.PARAM_ERROR, "同一层级下数据集名称已存在")

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

        await DatasetService._evict_all_cache(redis)

        return dataset_id

    @staticmethod
    async def _would_create_cycle(db: AsyncSession, dataset_id: int, new_parent_id: int) -> bool:
        if new_parent_id == 0:
            return False
        descendants = await dataset_repository.get_all_descendant_ids(db, dataset_id)
        return new_parent_id in descendants

    @staticmethod
    async def _update_children_tree_paths(
        db: AsyncSession,
        dataset_id: int,
        old_prefix: str,
        new_prefix: str,
    ):
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
        if not dataset_ids:
            raise BusinessException(ResultCode.PARAM_ERROR, "未指定要删除的数据集")

        total = len(dataset_ids)
        succeeded = 0
        failed = 0
        results = []

        for dataset_id in dataset_ids:
            try:
                dataset = await dataset_repository.get_by_id(db, dataset_id, with_deleted=True)
                if not dataset:
                    failed += 1
                    results.append({"datasetId": dataset_id, "status": "failed", "message": "数据集不存在"})
                    continue

                all_dataset_ids = await DatasetService._get_dataset_and_descendant_ids(db, dataset_id)
                all_datasets = await dataset_repository.get_all_datasets_for_tree_path_update(db, all_dataset_ids)

                children_map: dict[int, list[SysDataset]] = {}
                for ds in all_datasets:
                    pid = int(ds.parent_id)
                    if pid not in children_map:
                        children_map[pid] = []
                    children_map[pid].append(ds)

                leaf_ids = [ds_id for ds_id in all_dataset_ids if not children_map.get(ds_id)]

                for leaf_id in leaf_ids:
                    await DatasetItemService.delete_items_by_dataset(db, redis, leaf_id)

                depth_map = await dataset_repository.get_dataset_depth(db, all_dataset_ids)
                sorted_ids = sorted(all_dataset_ids, key=lambda x: depth_map.get(x, 0), reverse=True)
                await dataset_repository.delete_by_ids(db, sorted_ids)

                succeeded += 1
                results.append({"datasetId": dataset_id, "status": "success"})

            except Exception as e:
                failed += 1
                results.append({"datasetId": dataset_id, "status": "failed", "message": str(e)})

        await DatasetService._evict_all_cache(redis)

        return {
            "success": True,
            "message": f"删除完成：成功 {succeeded} 个，失败 {failed} 个",
            "data": {"total": total, "succeeded": succeeded, "failed": failed, "results": results},
        }

    @staticmethod
    async def _get_dataset_and_descendant_ids(db: AsyncSession, dataset_id: int) -> list[int]:
        descendants = await dataset_repository.get_all_descendant_ids(db, dataset_id)
        return [dataset_id] + descendants

    @staticmethod
    async def get_image_items(
        db: AsyncSession,
        redis: Redis,
        dataset_id: int,
        page_num: int = 1,
        page_size: int = 20,
        keywords: str | None = None,
    ) -> dict[str, Any]:
        leaf_ids = await dataset_repository.get_leaf_ids(db, dataset_id)
        total = await dataset_repository.get_items_count(db, leaf_ids, keywords)
        offset = (page_num - 1) * page_size
        items = await dataset_repository.get_items_paginated(db, leaf_ids, offset, page_size, keywords)

        if not items:
            return {"list": [], "total": total, "pageNum": page_num, "pageSize": page_size}

        item_ids = [int(item.id) for item in items]
        items_map, files_map = await dataset_repository.get_items_with_files_batch(db, item_ids)

        records = []
        for item in items:
            item_id = int(item.id)
            item_files = files_map.get(item_id, [])

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

            records.append({
                "id": item.id,
                "datasetId": item.dataset_id,
                "name": item.name,
                "createTime": format_time(item.create_time) if hasattr(item, "create_time") else None,
                "updateTime": format_time(item.update_time) if hasattr(item, "update_time") else None,
                "files": files,
                "imgUrl": image_urls,
            })

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
        dataset_id = data.get("datasetId")
        if not dataset_id:
            raise BusinessException(ResultCode.PARAM_ERROR, "数据集ID不能为空")

        dataset = await dataset_repository.get_by_id(db, dataset_id)
        if not dataset:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据集不存在")

        children_count = await dataset_repository.get_children_count(db, dataset_id)
        if children_count > 0:
            raise BusinessException(ResultCode.PARAM_ERROR, "不能在目录类型的数据集中创建数据项")

        dataset_item = SysDatasetItem(
            dataset_id=dataset_id,
            name=data.get("name", ""),
        )

        db.add(dataset_item)
        await db.flush()
        await db.refresh(dataset_item)

        await DatasetService._evict_all_cache(redis)

        return dataset_item.id

    @staticmethod
    async def get_item_detail(db: AsyncSession, item_id: int) -> dict[str, Any]:
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
        item = await dataset_repository.get_item_by_id(db, item_id)
        if not item:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据项不存在")

        if "name" in data:
            item.name = data["name"]

        item.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        await DatasetService._evict_all_cache(redis)

        return {"id": item_id}

    @staticmethod
    async def delete_dataset_item(
        db: AsyncSession,
        redis: Redis,
        item_id: int,
    ):
        item = await dataset_repository.get_item_by_id(db, item_id)
        if not item:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据项不存在")

        dataset_id = item.dataset_id
        await dataset_repository.delete_item_files_by_item_id(db, item_id)
        await dataset_repository.delete_item_by_id(db, item_id)

        await DatasetService._evict_all_cache(redis)

    @staticmethod
    async def delete_items_by_dataset(
        db: AsyncSession,
        redis: Redis,
        dataset_id: int,
    ) -> int:
        item_ids = await dataset_repository.get_item_ids_by_dataset(db, dataset_id)
        if not item_ids:
            return 0

        await dataset_repository.delete_item_files_by_item_ids(db, item_ids)
        await dataset_repository.delete_items_by_dataset_id(db, dataset_id)

        return len(item_ids)

    @staticmethod
    async def batch_delete_items(
        db: AsyncSession,
        redis: Redis,
        item_ids: list[int],
    ):
        if not item_ids:
            raise BusinessException(ResultCode.PARAM_ERROR, "未指定要删除的数据项")

        affected_dataset_ids: set[int] = set()
        for item_id in item_ids:
            item = await dataset_repository.get_item_by_id(db, item_id)
            if not item:
                continue
            affected_dataset_ids.add(int(item.dataset_id))
            await dataset_repository.delete_item_files_by_item_id(db, item_id)
            await dataset_repository.delete_item_by_id(db, item_id)

        await DatasetService._evict_all_cache(redis)

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
        if clear_file_content is None:
            raise BusinessException(ResultCode.PARAM_ERROR, "清晰图必须上传")
        if not hazy_files_data:
            raise BusinessException(ResultCode.PARAM_ERROR, "至少上传一张有雾图")

        dataset = await dataset_repository.get_by_id(db, dataset_id)
        if not dataset or dataset.deleted:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据集不存在")
        if dataset.type == "DIR":
            raise BusinessException(ResultCode.PARAM_ERROR, "目录类型数据集不允许创建数据项")

        item_name = name or f"Item_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        item = SysDatasetItem(dataset_id=dataset_id, name=item_name)
        db.add(item)
        await db.flush()
        await db.refresh(item)

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

        await DatasetService._evict_all_cache(redis)

        return await DatasetItemService.get_item_detail(db, item.id)

    @staticmethod
    async def batch_create_dataset_items_with_images(
        db: AsyncSession,
        redis: Redis,
        dataset_id: int,
        scene_type: str | None = None,
        files_data: list[dict] | None = None,
    ) -> dict:
        if not files_data:
            raise BusinessException(ResultCode.PARAM_ERROR, "至少上传一个文件")

        dataset = await dataset_repository.get_by_id(db, dataset_id)
        if not dataset or dataset.deleted:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据集不存在")
        if dataset.type == "DIR":
            raise BusinessException(ResultCode.PARAM_ERROR, "目录类型数据集不允许创建数据项")

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

        success_items = []
        failed_items = []
        total = len(groups)

        for prefix, files in groups.items():
            if not files["clear"]:
                failed_items.append({"fileName": prefix, "reason": f"未找到清晰图（需要 {prefix}_clear 或 {prefix}_gt 文件）"})
                continue
            if not files["hazy"]:
                failed_items.append({"fileName": prefix, "reason": f"未找到有雾图（需要 {prefix}_hazy 文件）"})
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
                success_items.append({"id": details["id"] if details else 0, "name": details.get("name"), "fileCount": file_count})
            except Exception as e:
                failed_items.append({"fileName": prefix, "reason": str(e)})

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
    async def get_item_file_detail(db: AsyncSession, file_id: int) -> dict[str, Any] | None:
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
        from app.service.file_service import FileService

        item = await dataset_repository.get_item_by_id(db, item_id)
        if not item:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据项不存在")

        valid_types = {"clear", "hazy", "depth", "segment"}
        if image_type not in valid_types:
            raise BusinessException(ResultCode.PARAM_ERROR, f"不支持的图片类型: {image_type}")

        if image_type == "hazy" and haze_level:
            valid_levels = {"light", "medium", "heavy"}
            if haze_level not in valid_levels:
                raise BusinessException(ResultCode.PARAM_ERROR, f"不支持的雾霾等级: {haze_level}")

        content = await file.read()
        if not file.filename:
            raise BusinessException(ResultCode.PARAM_ERROR, "文件名不能为空")

        file_info = await FileService.upload_file(
            db=db, filename=file.filename, content=content,
            content_type=file.content_type or "application/octet-stream",
        )

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

        await DatasetService._evict_all_cache(redis)

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
    async def update_item_file(db: AsyncSession, redis: Redis, file_id: int, data: dict[str, Any]):
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

        item = await dataset_repository.get_item_by_id(db, item_file.item_id)
        if item:
            await DatasetService._evict_all_cache(redis)

    @staticmethod
    async def delete_item_file(db: AsyncSession, redis: Redis, file_id: int):
        item_file = await dataset_repository.get_item_file_by_id(db, file_id)
        if not item_file:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "图片文件不存在")

        dataset_id = None
        item = await dataset_repository.get_item_by_id(db, item_file.item_id)
        if item:
            dataset_id = item.dataset_id

        await dataset_repository.delete_item_file_by_id(db, file_id)

        if dataset_id:
            await DatasetService._evict_all_cache(redis)

    @staticmethod
    async def batch_delete_item_files(db: AsyncSession, redis: Redis, file_ids: list[int]):
        if not file_ids:
            raise BusinessException(ResultCode.PARAM_ERROR, "未指定要删除的图片")

        affected_dataset_ids: set[int] = set()
        for fid in file_ids:
            item_file = await dataset_repository.get_item_file_by_id(db, fid)
            if not item_file:
                continue
            item = await dataset_repository.get_item_by_id(db, item_file.item_id)
            if item:
                affected_dataset_ids.add(int(item.dataset_id))

        await dataset_repository.delete_item_files_by_ids(db, file_ids)

        if affected_dataset_ids:
            await DatasetService._evict_all_cache(redis)
