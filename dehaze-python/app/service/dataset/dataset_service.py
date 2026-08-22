"""数据集聚合：树形结构 CRUD、环校验、统计聚合与缓存"""

import json
import re
import time
from typing import Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.base import get_current_user_id
from app.models.entity.sys_dataset import SysDataset
from app.repository.dataset_repository import dataset_repository
from app.repository.mongo_audit_log_repository import mongo_audit_log_repository
from app.service.dataset._shared import _build_file_vo, logger
from app.utils.datetime_utils import format_time

# XSS 危险模式：HTML 标签起始、javascript 协议、事件处理器（onXxx=）
_XSS_PATTERN = re.compile(
    r"<\s*/?\s*[a-zA-Z]|javascript:\s*|on\w+\s*=",
    re.IGNORECASE,
)

def _create_empty_stats() -> dict[str, Any]:
    return {
        "itemCount": 0,
        "fileCount": 0,
        "totalSize": 0,
        "annotatedCount": 0,
        "unannotatedCount": 0,
        "sceneDistribution": {},
        "hazeDistribution": {},
        "formatDistribution": {},
    }



def _merge_stats(parent: dict[str, Any], child: dict[str, Any]):
    parent["itemCount"] += child.get("itemCount", 0)
    parent["fileCount"] += child.get("fileCount", 0)
    parent["totalSize"] += child.get("totalSize", 0)
    parent["annotatedCount"] += child.get("annotatedCount", 0)
    parent["unannotatedCount"] += child.get("unannotatedCount", 0)
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
    def _validate_name_safety(name: str) -> None:
        """校验数据集名称安全性，拦截 XSS 攻击"""
        if name and _XSS_PATTERN.search(name):
            raise BusinessException(ResultCode.PARAM_ERROR, "数据集名称包含不安全的字符")

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
        except Exception as e:
            logger.warning(f"读取数据集缓存失败: {e}")

        datasets = await dataset_repository.find_all(db)

        try:
            serializable = []
            for ds in datasets:
                serializable.append(
                    {
                        "id": ds.id,
                        "parent_id": ds.parent_id,
                        "type": ds.type,
                        "name": ds.name,
                        "img": ds.img,
                        "description": ds.description,
                        "path": ds.path,
                        "size": ds.size,
                        "status": ds.status,
                        "deleted": ds.deleted,
                    }
                )
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
        except Exception as e:
            logger.warning(f"读取统计缓存失败: {e}")

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
                    stats_map[ds_id]["annotatedCount"] = st["annotatedCount"]
                    stats_map[ds_id]["unannotatedCount"] = st["unannotatedCount"]

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
        logger.debug(f"所有数据集统计信息计算完成，耗时 {cost_ms} ms，叶子节点 {len(leaf_ids)} 个")

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
        keyword: str | None = None,
        type: str | None = None,
        status: int | None = None,
    ) -> dict[str, Any]:
        root_datasets, total = await dataset_repository.find_root_page(
            db,
            page_num,
            page_size,
            keyword,
            type,
            status,
        )
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
            root_vo = DatasetService._entity_to_vo(
                root, root_stats, has_children_map.get(root_id, False)
            )

            children = direct_children_map.get(root_id, [])
            child_vos = []
            for child in children:
                cid = int(child.id)
                c_stats = stats_map.get(cid, _create_empty_stats())
                child_vos.append(
                    DatasetService._entity_to_vo(child, c_stats, has_children_map.get(cid, False))
                )
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
            child_vo = DatasetService._entity_to_vo(
                child, c_stats, has_children_map.get(cid, False)
            )
            child_vo["children"] = []
            result.append(child_vo)

        return result

    @staticmethod
    async def get_dataset_options(db: AsyncSession, redis: Redis) -> list[dict[str, Any]]:
        try:
            cached = await redis.get(DatasetService.CACHE_OPTIONS_KEY)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"读取选项缓存失败: {e}")

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
    async def get_evaluation_options(
        db: AsyncSession,
        task_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """获取测试集选项（算法评估接入）。

        仅返回含清晰图 GT（type=clear）且启用的数据集，按 taskType 过滤（T-DS-046~048）。
        返回扁平 label-value 列表。
        """
        datasets = await dataset_repository.find_datasets_with_clear_gt(db, task_type)
        return [
            {"value": int(ds.id), "label": ds.name}
            for ds in datasets
        ]

    @staticmethod
    async def get_dataset_by_id(
        db: AsyncSession,
        redis: Redis,
        dataset_id: int,
    ) -> dict[str, Any] | None:
        dataset = await dataset_repository.get_by_id(db, dataset_id)
        if not dataset:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据集不存在")

        stats_map = await DatasetService.get_all_dataset_stats(db, redis)
        statistics = stats_map.get(int(dataset_id), _create_empty_stats())

        return {
            "id": dataset.id,
            "parentId": dataset.parent_id,
            "type": dataset.type,
            "name": dataset.name,
            "img": dataset.img,
            "description": dataset.description,
            "path": dataset.path,
            "size": dataset.size,
            "status": dataset.status,
            "createTime": format_time(dataset.create_time),
            "updateTime": format_time(dataset.update_time),
            "statistics": statistics,
        }

    @staticmethod
    async def create_dataset(
        db: AsyncSession,
        redis: Redis,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        parent_id = data.get("parentId", 0)
        name = data.get("name", "")

        DatasetService._validate_name_safety(name)

        if parent_id != 0:
            parent = await dataset_repository.get_by_id(db, parent_id)
            if not parent:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "父数据集不存在")

        if name:
            exists = await dataset_repository.check_name_exists(db, parent_id, name)
            if exists:
                raise BusinessException(ResultCode.PARAM_ERROR, "同一层级下数据集名称已存在")

        dataset = SysDataset(
            parent_id=parent_id,
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
    async def update_dataset(
        db: AsyncSession,
        redis: Redis,
        dataset_id: int,
        data: dict[str, Any],
    ) -> dict[str, Any]:
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

            dataset.parent_id = new_parent_id

        if "name" in data and data["name"] != dataset.name:
            DatasetService._validate_name_safety(data["name"])
            check_parent = new_parent_id if new_parent_id is not None else old_parent_id
            exists = await dataset_repository.check_name_exists(
                db,
                check_parent,
                data["name"],
                exclude_id=dataset_id,
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

        await DatasetService._evict_all_cache(redis)

        return await DatasetService.get_dataset_by_id(db, redis, dataset_id)

    @staticmethod
    async def _would_create_cycle(db: AsyncSession, dataset_id: int, new_parent_id: int) -> bool:
        if new_parent_id == 0:
            return False
        descendants = await dataset_repository.get_all_descendant_ids(db, dataset_id)
        return new_parent_id in descendants

    @staticmethod
    async def delete_dataset(
        db: AsyncSession,
        redis: Redis,
        dataset_id: int,
    ) -> None:
        """删除单个数据集（匹配 Java deleteDataset 行为：不存在时抛异常，成功返回 void）"""
        dataset = await dataset_repository.get_by_id(db, dataset_id)
        if not dataset:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据集不存在")
        await DatasetService.delete_datasets(db, redis, [dataset_id])

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

        # 1. 批量预查询数据集存在性（1 次 IN 查询，替代 N 次 get_by_id）
        existing_datasets = await dataset_repository.get_by_ids(db, dataset_ids, with_deleted=True)
        existing_map = {int(d.id): d for d in existing_datasets}

        # 分类存在/不存在
        valid_dataset_ids: list[int] = []
        for dataset_id in dataset_ids:
            if dataset_id not in existing_map:
                failed += 1
                results.append(
                    {
                        "id": dataset_id,
                        "status": "failed",
                        "message": "数据集不存在",
                        "errorCode": "RESOURCE_NOT_FOUND",
                    }
                )
            else:
                valid_dataset_ids.append(dataset_id)

        if valid_dataset_ids:
            try:
                # 2. 批量获取所有后代 ID（1 次全表查询 + 内存 BFS，
                #    替代 N 次 _get_dataset_and_descendant_ids）
                descendants_map = await dataset_repository.get_all_descendant_ids_batch(
                    db, valid_dataset_ids
                )

                # 3. 收集所有需要删除的数据集 ID（去重）
                all_ids_set: set[int] = set()
                for dataset_id in valid_dataset_ids:
                    all_ids_set.update(descendants_map.get(dataset_id, [dataset_id]))
                unique_ids_to_delete = list(all_ids_set)

                # 4. 批量查询所有待删除数据集，构建 children_map 用于识别叶子节点
                all_datasets = await dataset_repository.get_datasets_by_ids(
                    db, unique_ids_to_delete
                )
                children_map: dict[int, list[SysDataset]] = {}
                for ds in all_datasets:
                    pid = int(ds.parent_id)
                    if pid not in children_map:
                        children_map[pid] = []
                    children_map[pid].append(ds)

                # 5. 批量识别叶子节点（待删除集合中没有子节点的）
                all_leaf_ids = [
                    ds_id for ds_id in unique_ids_to_delete if not children_map.get(ds_id)
                ]

                # 6. 批量删除所有叶子节点下的数据项
                # （1 次查 item_ids + 1 次删 files + 1 次删 items，
                #   替代 N 次 delete_items_by_dataset）
                if all_leaf_ids:
                    all_item_ids = await dataset_repository.get_item_ids_by_dataset_ids(
                        db, all_leaf_ids
                    )
                    if all_item_ids:
                        await dataset_repository.delete_item_files_by_item_ids(db, all_item_ids)
                        await dataset_repository.delete_items_by_ids(db, all_item_ids)

                # 7. 批量删除所有数据集（1 次物理删除，替代 N 次 delete_by_ids）
                await dataset_repository.delete_by_ids(db, unique_ids_to_delete)

                # 8. 记录成功结果
                for dataset_id in valid_dataset_ids:
                    succeeded += 1
                    results.append({"id": dataset_id, "status": "success"})

            except Exception as e:
                # 批量删除失败，回滚并标记所有有效数据集为失败
                await db.rollback()
                for dataset_id in valid_dataset_ids:
                    failed += 1
                    results.append(
                        {
                            "id": dataset_id,
                            "status": "failed",
                            "message": str(e),
                            "errorCode": "SYSTEM_ERROR",
                        }
                    )

        await DatasetService._evict_all_cache(redis)

        mongo_audit_log_repository.create_audit_async(
            operator_id=get_current_user_id(),
            target_type="dataset",
            target_id=dataset_ids,
            action="delete",
            module="dataset",
        )

        return {
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "results": results,
        }

    @staticmethod
    async def get_image_items(
        db: AsyncSession,
        redis: Redis,
        dataset_id: int | None,
        page_num: int = 1,
        page_size: int = 20,
        keywords: str | None = None,
        scene_type: str | None = None,
    ) -> dict[str, Any]:
        if dataset_id:
            leaf_ids = await dataset_repository.get_leaf_ids(db, dataset_id)
        else:
            leaf_ids = []
        total = await dataset_repository.get_items_count(db, leaf_ids, keywords)
        offset = (page_num - 1) * page_size
        items = await dataset_repository.get_items_paginated(
            db, leaf_ids, offset, page_size, keywords
        )

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
            clear_image = None
            hazy_images = []
            for item_file, file_obj in item_files:
                file_vo = _build_file_vo(item_file, file_obj)
                files.append(file_vo)
                if file_obj is not None:
                    # image_urls 简化对齐 _build_file_vo 的 url 字段
                    image_urls.append(
                        {
                            "id": file_obj.id,
                            "type": item_file.type,
                            "url": file_vo["url"],
                            "thumbnailUrl": file_vo["url"],
                        }
                    )
                if item_file.type == "clear" and clear_image is None:
                    clear_image = file_vo
                elif item_file.type == "hazy":
                    hazy_images.append(file_vo)

            records.append(
                {
                    "id": item.id,
                    "datasetId": item.dataset_id,
                    "name": item.name,
                    "createTime": format_time(item.create_time)
                    if hasattr(item, "create_time")
                    else None,
                    "updateTime": format_time(item.update_time)
                    if hasattr(item, "update_time")
                    else None,
                    "files": files,
                    "imgUrl": image_urls,
                    "clearImage": clear_image,
                    "hazyImages": hazy_images,
                }
            )

        return {
            "list": records,
            "total": total,
            "pageNum": page_num,
            "pageSize": page_size,
        }


