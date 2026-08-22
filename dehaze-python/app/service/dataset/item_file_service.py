"""条目文件聚合：文件上传绑定与查询"""

from typing import Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_dataset import SysItemFile
from app.repository.dataset_repository import dataset_repository
from app.service.dataset._shared import _build_file_vo
from app.service.dataset.dataset_service import DatasetService
from app.service.file_service import FileService


class ItemFileService:
    """图片文件服务"""

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

        item = await dataset_repository.get_item_by_id(db, item_id)
        if not item:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据项不存在")

        # type 支持 clear/hazy/trans/depth/segment，不做硬性枚举校验
        # haze_level 支持多种规范（light/medium/heavy、beta=X、A=X,beta=Y 等），可为空

        content = await file.read()
        if not file.filename:
            raise BusinessException(ResultCode.PARAM_ERROR, "文件名不能为空")

        file_info = await FileService.upload_file(
            db=db,
            filename=file.filename,
            content=content,
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

        return _build_file_vo(item_file, file_info)

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
                failure_details.append(
                    {
                        "identifier": str(fid),
                        "reason": "图片文件不存在",
                    }
                )

        # 批量查询受影响的数据集 ID（避免 N+1）
        affected_dataset_ids: set[int] = set()
        if success_ids:
            # 从已查询的 item_files 中提取 item_id，批量查询 items 获取 dataset_id
            affected_item_ids = {int(f.item_id) for f in existing_item_files}
            if affected_item_ids:
                affected_items = await dataset_repository.get_items_by_ids(
                    db, list(affected_item_ids)
                )
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

