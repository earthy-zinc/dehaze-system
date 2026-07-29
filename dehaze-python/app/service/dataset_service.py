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
from app.models.base import get_current_user_id
from app.models.entity.sys_dataset import (SysDataset, SysDatasetItem,
                                           SysItemFile)
from app.repository.dataset_repository import dataset_repository
from app.repository.mongo_audit_log_repository import mongo_audit_log_repository
from app.service.file_service import FileService
from app.utils.datetime_utils import format_time
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# XSS 危险模式：HTML 标签起始、javascript 协议、事件处理器（onXxx=）
_XSS_PATTERN = re.compile(
    r'<\s*/?\s*[a-zA-Z]|javascript:\s*|on\w+\s*=',
    re.IGNORECASE,
)


def _extract_file_prefix(filename: str) -> str:
    """提取文件名前导数字作为分组键（如 01_GT.png → "01"，1000_1_0.74905.png → "1000"）。
    无前导数字时返回完整 stem（去除扩展名）。"""
    name = re.sub(r'\.[^.]+$', '', filename)
    match = re.match(r'^(\d+)', name)
    if match:
        return match.group(1)
    return name


def _is_clear_image(filename: str) -> bool:
    """判断文件名是否为清晰图（含 clear/gt/GT/clean 关键字）"""
    name_lower = filename.lower()
    return any(kw in name_lower for kw in ('clear', '_gt', 'gt.', 'clean'))


def _is_hazy_image(filename: str) -> bool:
    """判断文件名是否为有雾图（含 hazy/haze 关键字）"""
    name_lower = filename.lower()
    return 'hazy' in name_lower or 'haze' in name_lower


def _is_trans_image(filename: str) -> bool:
    """判断文件名是否为透射率图（含 trans/Transmission 关键字）"""
    name_lower = filename.lower()
    return 'trans' in name_lower


def _extract_haze_level(filename: str) -> str:
    """从有雾图文件名提取雾霾程度，支持多种规范。
    无法解析时返回空字符串（表示未标注）。

    支持格式：
    - _hazy_light / _hazy_medium / _hazy_heavy → light/medium/heavy
    - {id}_{idx}_{beta}.png（如 1000_1_0.74905.png）→ beta=0.74905
    - {id}_{A}_{beta}.jpg（如 0025_0.8_0.2.jpg）→ beta=0.2（无法可靠区分 A 和 idx，统一取最后一个数值作为 beta）
    - 无参数后缀（如 01_hazy.png）→ 空字符串
    """
    name = re.sub(r'\.[^.]+$', '', filename)

    # 1. 人工分级：_hazy_light / _hazy_medium / _hazy_heavy
    match = re.search(r'_hazy_(light|medium|heavy)', filename, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # 2. 学术参数格式：{id}_{idx}_{beta} 或 {id}_{A}_{beta} 等
    #    统一取最后一个数值作为 beta（无法可靠区分 A 和 idx）
    parts = name.split('_')
    if len(parts) >= 3:
        try:
            num_parts = []
            for p in parts[1:]:  # 跳过第一段（id）
                try:
                    num_parts.append(float(p))
                except ValueError:
                    continue
            if num_parts:
                beta = num_parts[-1]
                return f"beta={beta}"
        except (ValueError, IndexError):
            pass

    return ""


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


def _build_file_vo(item_file, file_obj) -> dict[str, Any]:
    """构建图片文件 VO（统一字段命名，对齐 SDK ImageUrlVO）。
    用于 ItemFileService 和 DatasetItemService 的所有文件响应。"""
    # 从文件名提取格式（扩展名），统一返回小写
    file_format = None
    if file_obj and file_obj.name and "." in file_obj.name:
        file_format = file_obj.name.rsplit(".", 1)[-1].lower()
    elif file_obj and file_obj.type:
        file_format = file_obj.type.lower()

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
        "fileName": file_obj.name if file_obj else None,
        "name": file_obj.name if file_obj else None,
        "sizeBytes": file_obj.size_bytes if file_obj else None,
        "size": file_obj.size_bytes if file_obj else None,
        "formattedSize": file_obj.size if file_obj else None,
        "format": file_format,
        "md5": file_obj.md5 if file_obj else None,
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
                serializable.append({
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
            db, page_num, page_size, keyword, type, status,
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
        existing_datasets = await dataset_repository.get_by_ids(
            db, dataset_ids, with_deleted=True)
        existing_map = {int(d.id): d for d in existing_datasets}

        # 分类存在/不存在
        valid_dataset_ids: list[int] = []
        for dataset_id in dataset_ids:
            if dataset_id not in existing_map:
                failed += 1
                results.append({
                    "id": dataset_id,
                    "status": "failed",
                    "message": "数据集不存在",
                    "errorCode": "RESOURCE_NOT_FOUND",
                })
            else:
                valid_dataset_ids.append(dataset_id)

        if valid_dataset_ids:
            try:
                # 2. 批量获取所有后代 ID（1 次全表查询 + 内存 BFS，替代 N 次 _get_dataset_and_descendant_ids）
                descendants_map = await dataset_repository.get_all_descendant_ids_batch(
                    db, valid_dataset_ids)

                # 3. 收集所有需要删除的数据集 ID（去重）
                all_ids_set: set[int] = set()
                for dataset_id in valid_dataset_ids:
                    all_ids_set.update(descendants_map.get(dataset_id, [dataset_id]))
                unique_ids_to_delete = list(all_ids_set)

                # 4. 批量查询所有待删除数据集，构建 children_map 用于识别叶子节点
                all_datasets = await dataset_repository.get_datasets_by_ids(
                    db, unique_ids_to_delete)
                children_map: dict[int, list[SysDataset]] = {}
                for ds in all_datasets:
                    pid = int(ds.parent_id)
                    if pid not in children_map:
                        children_map[pid] = []
                    children_map[pid].append(ds)

                # 5. 批量识别叶子节点（待删除集合中没有子节点的）
                all_leaf_ids = [
                    ds_id for ds_id in unique_ids_to_delete
                    if not children_map.get(ds_id)
                ]

                # 6. 批量删除所有叶子节点下的数据项
                # （1 次查 item_ids + 1 次删 files + 1 次删 items，替代 N 次 delete_items_by_dataset）
                if all_leaf_ids:
                    all_item_ids = await dataset_repository.get_item_ids_by_dataset_ids(
                        db, all_leaf_ids)
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
                    results.append({
                        "id": dataset_id,
                        "status": "failed",
                        "message": str(e),
                        "errorCode": "SYSTEM_ERROR",
                    })

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
            clear_image = None
            hazy_images = []
            for item_file, file_obj in item_files:
                file_vo = _build_file_vo(item_file, file_obj)
                files.append(file_vo)
                if file_obj is not None:
                    image_urls.append({
                        "id": file_obj.id,
                        "type": item_file.type,
                        "url": file_obj.url,
                        "thumbnailUrl": file_obj.url,
                    })
                if item_file.type == "clear" and clear_image is None:
                    clear_image = file_vo
                elif item_file.type == "hazy":
                    hazy_images.append(file_vo)

            records.append({
                "id": item.id,
                "datasetId": item.dataset_id,
                "name": item.name,
                "createTime": format_time(item.create_time) if hasattr(item, "create_time") else None,
                "updateTime": format_time(item.update_time) if hasattr(item, "update_time") else None,
                "files": files,
                "imgUrl": image_urls,
                "clearImage": clear_image,
                "hazyImages": hazy_images,
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
    ) -> dict[str, Any]:
        dataset_id = data.get("datasetId")
        if not dataset_id:
            raise BusinessException(ResultCode.PARAM_ERROR, "数据集ID不能为空")

        dataset = await dataset_repository.get_by_id(db, dataset_id)
        if not dataset:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据集不存在")

        children_count = await dataset_repository.get_children_count(db, dataset_id)
        if children_count > 0:
            raise BusinessException(ResultCode.PARAM_ERROR, "不能在目录类型的数据集中创建数据项")

        item_name = data.get("name", "")
        dataset_item = SysDatasetItem(
            dataset_id=dataset_id,
            name=item_name,
        )

        db.add(dataset_item)
        await db.flush()
        await db.refresh(dataset_item)

        await DatasetService._evict_all_cache(redis)

        return {
            "id": dataset_item.id,
            "datasetId": dataset_item.dataset_id,
            "name": dataset_item.name,
        }

    @staticmethod
    def _build_item_file_vo(item_file, file_obj) -> dict[str, Any]:
        """构建数据项详情中的文件 VO（对齐 SDK ImageUrlVO）。
        委托给模块级 _build_file_vo，保留方法签名以兼容现有调用。"""
        return _build_file_vo(item_file, file_obj)

    @staticmethod
    async def get_item_detail(db: AsyncSession, item_id: int) -> dict[str, Any]:
        item, item_files = await dataset_repository.get_item_with_files(db, item_id)
        if not item:
            return {}

        files = []
        image_urls = []
        clear_image = None
        hazy_images = []
        for item_file, file_obj in item_files:
            file_vo = _build_file_vo(item_file, file_obj)
            files.append(file_vo)
            if file_obj is not None:
                image_urls.append({
                    "id": file_obj.id,
                    "type": item_file.type,
                    "url": file_obj.url,
                    "thumbnailUrl": file_obj.url,
                })
            # 按类型拆分：clearImage / hazyImages（对齐 SDK DatasetItemVO）
            if item_file.type == "clear" and clear_image is None:
                clear_image = file_vo
            elif item_file.type == "hazy":
                hazy_images.append(file_vo)

        return {
            "id": item.id,
            "datasetId": item.dataset_id,
            "name": item.name,
            "createTime": format_time(item.create_time) if hasattr(item, "create_time") else None,
            "updateTime": format_time(item.update_time) if hasattr(item, "update_time") else None,
            "files": files,
            "imgUrl": image_urls,
            "clearImage": clear_image,
            "hazyImages": hazy_images,
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

        await DatasetService._evict_all_cache(redis)

        return {
            "id": item.id,
            "datasetId": item.dataset_id,
            "name": item.name,
        }

    @staticmethod
    async def delete_dataset_item(
        db: AsyncSession,
        redis: Redis,
        item_id: int,
    ):
        item = await dataset_repository.get_item_by_id(db, item_id)
        if not item:
            return

        await dataset_repository.delete_item_files_by_item_id(db, item_id)
        await dataset_repository.delete_item_by_id(db, item_id)

        await DatasetService._evict_all_cache(redis)

    @staticmethod
    async def batch_delete_items(
        db: AsyncSession,
        redis: Redis,
        item_ids: list[int],
    ) -> dict[str, Any]:
        if not item_ids:
            raise BusinessException(ResultCode.PARAM_ERROR, "未指定要删除的数据项")

        # 批量查询存在的数据项（避免 N+1）
        existing_items = await dataset_repository.get_items_by_ids(db, item_ids)
        existing_ids_set = {int(item.id) for item in existing_items}
        success_ids: list[int] = []
        failure_details: list[dict[str, str]] = []

        for item_id in item_ids:
            if item_id in existing_ids_set:
                success_ids.append(item_id)
            else:
                failure_details.append({
                    "identifier": str(item_id),
                    "reason": "数据项不存在",
                })

        # 批量删除关联文件和数据项（2 条 SQL，替代 2N 条）
        if success_ids:
            await dataset_repository.delete_item_files_by_item_ids(db, success_ids)
            await dataset_repository.delete_items_by_ids(db, success_ids)

        await DatasetService._evict_all_cache(redis)

        mongo_audit_log_repository.create_audit_async(
            operator_id=get_current_user_id(),
            target_type="dataset_item",
            target_id=item_ids,
            action="delete",
            module="dataset",
        )

        return {
            "successCount": len(success_ids),
            "failedCount": len(failure_details),
            "message": f"批量删除完成: 成功{len(success_ids)}个, 失败{len(failure_details)}个",
            "successIds": success_ids,
            "failureDetails": failure_details,
        }

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
        # 清晰图和有雾图均为可选（适配不同数据集规范：GT+Hazy 配对型、仅 Hazy 无 GT 型等）
        if clear_file_content is None and not hazy_files_data:
            raise BusinessException(ResultCode.PARAM_ERROR, "至少上传一张图片（清晰图或有雾图）")

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

        # 清晰图（可选）
        if clear_file_content is not None:
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

        # 有雾图（可选，haze_level 支持多种规范：light/medium/heavy、beta=X、A=X,beta=Y 等）
        for hfd in (hazy_files_data or []):
            haze_level = hfd.get("hazeLevel", "")
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
            trans = _is_trans_image(filename)

            if not clear and not hazy and not trans:
                unpaired.append({"fileName": filename, "reason": "无法识别文件类型，文件名需包含 clear/gt/clean、hazy/haze 或 trans 关键字"})
                continue

            prefix = _extract_file_prefix(filename)
            if not prefix:
                unpaired.append({"fileName": filename, "reason": "无法提取文件名前缀"})
                continue

            if prefix not in groups:
                groups[prefix] = {"clear": [], "hazy": [], "trans": []}

            if trans:
                groups[prefix]["trans"].append(fd)
            elif clear:
                groups[prefix]["clear"].append(fd)
            elif hazy:
                haze_level = _extract_haze_level(filename)
                fd["hazeLevel"] = haze_level
                groups[prefix]["hazy"].append(fd)

        success_items = []
        failed_items = []
        # total 为上传的文件总数（对齐 SDK BatchUploadResultVO.total = 总文件数）
        total = len(files_data)

        for prefix, files in groups.items():
            # 清晰图和有雾图均为可选（适配不同数据集规范）
            if not files["clear"] and not files["hazy"]:
                failed_items.append({"fileName": prefix, "reason": "未找到任何可识别的图片"})
                continue

            try:
                clear_fd = files["clear"][0] if files["clear"] else None
                details = await DatasetItemService.upload_dataset_item_with_images(
                    db=db,
                    redis=redis,
                    dataset_id=dataset_id,
                    name=prefix,
                    scene_type=scene_type,
                    clear_file_content=clear_fd["content"] if clear_fd else None,
                    clear_filename=clear_fd["filename"] if clear_fd else "",
                    clear_content_type=clear_fd.get("contentType", "application/octet-stream") if clear_fd else "",
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
    def _build_file_vo(item_file, file_obj) -> dict[str, Any]:
        """构建图片文件 VO（委托给模块级 _build_file_vo）"""
        return _build_file_vo(item_file, file_obj)

    @staticmethod
    async def get_item_file_detail(db: AsyncSession, file_id: int) -> dict[str, Any] | None:
        result = await dataset_repository.get_item_file_with_file(db, file_id)
        if not result:
            return None

        item_file, file_obj = result
        return _build_file_vo(item_file, file_obj)

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

        # type 支持 clear/hazy/trans/depth/segment，不做硬性枚举校验
        # haze_level 支持多种规范（light/medium/heavy、beta=X、A=X,beta=Y 等），可为空

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
            scene_type=scene_type or "",
            haze_level=haze_level or "",
            description=description,
        )
        db.add(item_file)
        await db.flush()
        await db.refresh(item_file)

        await DatasetService._evict_all_cache(redis)

        return ItemFileService._build_file_vo(item_file, file_info)

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

        # 批量查询存在的图片文件记录（避免 N+1）
        existing_item_files = await dataset_repository.get_item_files_by_ids(db, file_ids)
        existing_ids_set = {int(f.id) for f in existing_item_files}
        success_ids: list[int] = []
        failure_details: list[dict[str, str]] = []

        for fid in file_ids:
            if fid in existing_ids_set:
                success_ids.append(fid)
            else:
                failure_details.append({
                    "identifier": str(fid),
                    "reason": "图片文件不存在",
                })

        # 批量查询受影响的数据集 ID（避免 N+1）
        affected_dataset_ids: set[int] = set()
        if success_ids:
            # 从已查询的 item_files 中提取 item_id，批量查询 items 获取 dataset_id
            affected_item_ids = {int(f.item_id) for f in existing_item_files}
            if affected_item_ids:
                affected_items = await dataset_repository.get_items_by_ids(
                    db, list(affected_item_ids))
                for item in affected_items:
                    affected_dataset_ids.add(int(item.dataset_id))

            await dataset_repository.delete_item_files_by_ids(db, success_ids)

        if affected_dataset_ids:
            await DatasetService._evict_all_cache(redis)

        return {
            "successCount": len(success_ids),
            "failedCount": len(failure_details),
            "message": f"批量删除完成: 成功{len(success_ids)}个, 失败{len(failure_details)}个",
            "successIds": success_ids,
            "failureDetails": failure_details,
        }
