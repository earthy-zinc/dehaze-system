"""
图片文件路由

基础路径: /api/v1/item-files
"""
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.common import BatchDeleteForm
from app.models.schema.dataset import (BatchOperationResultVO,
                                       ItemFileUpdateForm, ItemFileVO)
from app.service.dataset_service import ItemFileService
from fastapi import (APIRouter, Body, Depends, File, Form, Path,
                     UploadFile)
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(
    prefix="/api/v1/item-files",
    tags=["图片文件管理"],
    dependencies=[Depends(get_current_user)],
)


@router.post("", response_model=Result[ItemFileVO], summary="上传数据项图片")
async def upload_item_file(
    file: UploadFile = File(..., description="图片文件"),
    itemId: int = Form(..., description="所属数据项ID"),
    type: str = Form(..., description="图片类型(clear/hazy/depth/segment)"),
    sceneType: str = Form(default="", description="场景类型"),
    hazeLevel: str = Form(default="", description="雾霾等级(light/medium/heavy)"),
    description: str = Form(default="", description="描述"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    result = await ItemFileService.upload_item_file(
        db=db,
        redis=redis,
        item_id=itemId,
        image_type=type,
        scene_type=sceneType,
        haze_level=hazeLevel,
        description=description,
        file=file,
    )
    return success(result, "上传成功")


@router.delete("/batch", response_model=Result[BatchOperationResultVO], summary="批量删除图片")
async def batch_delete_item_files(
    body: BatchDeleteForm = Body(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    result = await ItemFileService.batch_delete_item_files(db, redis, body.ids)
    return success(result, "删除成功")


@router.get("/{file_id}", response_model=Result[ItemFileVO], summary="获取图片详细信息")
async def get_item_file(
    file_id: int = Path(..., description="图片文件关联ID"),
    db: AsyncSession = Depends(get_db),
):
    detail = await ItemFileService.get_item_file_detail(db, file_id)
    if not detail:
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "图片文件不存在")
    return success(detail)


@router.put("/{file_id}", response_model=Result[None], summary="修改图片信息")
async def update_item_file(
    file_id: int = Path(..., description="图片文件关联ID"),
    body: ItemFileUpdateForm = Body(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    await ItemFileService.update_item_file(
        db, redis, file_id, body.model_dump(exclude_none=True),
    )
    return success(msg="更新成功")


@router.delete("/{file_id}", response_model=Result[None], summary="删除图片")
async def delete_item_file(
    file_id: int = Path(..., description="图片文件关联ID"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    await ItemFileService.delete_item_file(db, redis, file_id)
    return success(msg="删除成功")
