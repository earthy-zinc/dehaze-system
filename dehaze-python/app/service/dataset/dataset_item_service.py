"""数据集条目聚合：条目 CRUD 与导入文件名解析"""

import io
import re
from datetime import datetime
from typing import Any

import PIL.Image
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.base import get_current_user_id
from app.models.entity.sys_dataset import SysDatasetItem, SysItemFile
from app.repository.dataset_repository import dataset_repository
from app.repository.mongo_audit_log_repository import mongo_audit_log_repository
from app.service.dataset._shared import _build_file_vo
from app.service.dataset.dataset_service import DatasetService
from app.service.file_service import FileService
from app.utils.datetime_utils import format_time


def _extract_file_prefix(filename: str) -> str:
    """提取文件名前导数字作为分组键（如 01_GT.png → "01"，1000_1_0.74905.png → "1000"）。
    无前导数字时返回完整 stem（去除扩展名）。"""
    name = re.sub(r"\.[^.]+$", "", filename)
    match = re.match(r"^(\d+)", name)
    if match:
        return match.group(1)
    return name


def _is_clear_image(filename: str) -> bool:
    """判断文件名是否为清晰图（含 clear/gt/GT/clean 关键字）"""
    name_lower = filename.lower()
    return any(kw in name_lower for kw in ("clear", "_gt", "gt.", "clean"))


def _is_hazy_image(filename: str) -> bool:
    """判断文件名是否为有雾图（含 hazy/haze 关键字）"""
    name_lower = filename.lower()
    return "hazy" in name_lower or "haze" in name_lower


def _is_trans_image(filename: str) -> bool:
    """判断文件名是否为透射率图（含 trans/Transmission 关键字）"""
    name_lower = filename.lower()
    return "trans" in name_lower


def _extract_haze_level(filename: str) -> str:
    """从有雾图文件名提取雾霾程度，支持多种规范。
    无法解析时返回空字符串（表示未标注）。

    支持格式：
    - _hazy_light / _hazy_medium / _hazy_heavy → light/medium/heavy
    - {id}_{idx}_{beta}.png（如 1000_1_0.74905.png）→ beta=0.74905
    - {id}_{A}_{beta}.jpg（如 0025_0.8_0.2.jpg）→ beta=0.2（无法可靠区分 A 和 idx，
      统一取最后一个数值作为 beta）
    - 无参数后缀（如 01_hazy.png）→ 空字符串
    """
    name = re.sub(r"\.[^.]+$", "", filename)

    # 1. 人工分级：_hazy_light / _hazy_medium / _hazy_heavy
    match = re.search(r"_hazy_(light|medium|heavy)", filename, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # 2. 学术参数格式：{id}_{idx}_{beta} 或 {id}_{A}_{beta} 等
    #    统一取最后一个数值作为 beta（无法可靠区分 A 和 idx）
    parts = name.split("_")
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
                image_urls.append(
                    {
                        "id": file_obj.id,
                        "type": item_file.type,
                        "url": file_vo["url"],
                        "thumbnailUrl": file_vo["url"],
                    }
                )
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
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据项不存在")

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
                failure_details.append(
                    {
                        "identifier": str(item_id),
                        "reason": "数据项不存在",
                    }
                )

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

        # 校验配对图片分辨率一致性（清晰图存在时才校验，对齐 Java/Go 实现）
        clear_dims = None
        if clear_file_content is not None:
            try:
                with PIL.Image.open(io.BytesIO(clear_file_content)) as img:
                    clear_dims = img.size
            except Exception:
                raise BusinessException(
                    ResultCode.PARAM_ERROR, "清晰图文件格式错误或无法解析"
                ) from None
            for hfd in hazy_files_data or []:
                try:
                    with PIL.Image.open(io.BytesIO(hfd["content"])) as img:
                        hazy_dims = img.size
                except Exception:
                    raise BusinessException(
                        ResultCode.PARAM_ERROR,
                        f"有雾图 {hfd.get('filename', '')} 格式错误或无法解析",
                    ) from None
                if hazy_dims[0] != clear_dims[0] or hazy_dims[1] != clear_dims[1]:
                    raise BusinessException(
                        ResultCode.PARAM_ERROR,
                        f"配对图片分辨率不一致，清晰图：{clear_dims[0]}x{clear_dims[1]}，"
                        f"有雾图 {hfd.get('filename', '')}：{hazy_dims[0]}x{hazy_dims[1]}",
                    )

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
                db,
                clear_filename,
                clear_file_content,
                clear_content_type,
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
        for hfd in hazy_files_data or []:
            haze_level = hfd.get("hazeLevel", "")
            hazy_sys_file = await FileService.upload_file(
                db,
                hfd["filename"],
                hfd["content"],
                hfd.get("contentType", "application/octet-stream"),
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
                unpaired.append(
                    {
                        "fileName": filename,
                        "reason": (
                            "无法识别文件类型，文件名需包含 clear/gt/clean、"
                            "hazy/haze 或 trans 关键字"
                        ),
                    }
                )
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
                    clear_content_type=clear_fd.get("contentType", "application/octet-stream")
                    if clear_fd
                    else "",
                    hazy_files_data=files["hazy"],
                )
                file_count = len(details.get("files", [])) if details else 0
                success_items.append(
                    {
                        "id": details["id"] if details else 0,
                        "name": details.get("name"),
                        "fileCount": file_count,
                    }
                )
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


