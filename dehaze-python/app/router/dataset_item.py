"""
数据项路由

基础路径: /api/v1/dataset-items
"""
from typing import Optional

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.common import BatchDeleteForm
from app.models.schema.dataset import (BatchOperationResultVO,
                                       BatchUploadResultVO,
                                       DatasetItemCreateForm, DatasetItemVO,
                                       DatasetItemPageVO,
                                       DatasetItemUpdateForm)
from app.service.dataset_service import DatasetItemService, DatasetService
from fastapi import (APIRouter, Body, Depends, File, Form, Path, Query,
                     UploadFile)
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(
    prefix="/api/v1/dataset-items",
    tags=["数据项管理"],
    dependencies=[Depends(get_current_user)],
)


@router.get("", response_model=Result[DatasetItemPageVO], summary="分页查询数据项列表")
async def list_dataset_items(
    datasetId: Optional[int] = Query(default=None, description="所属数据集ID"),
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=20, ge=1, le=100, description="每页数量"),
    keywords: Optional[str] = Query(default=None, description="搜索关键词"),
    sceneType: Optional[str] = Query(default=None, description="场景类型筛选"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    result = await DatasetService.get_image_items(
        db, redis, datasetId, pageNum, pageSize, keywords, sceneType,
    )
    return success(result)


@router.get("/{item_id}", response_model=Result[DatasetItemVO], summary="获取数据项详情")
async def get_dataset_item(
    item_id: int = Path(..., description="数据项ID"),
    db: AsyncSession = Depends(get_db),
):
    detail = await DatasetItemService.get_item_detail(db, item_id)
    if not detail:
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据项不存在")
    return success(detail)


@router.post("", response_model=Result[DatasetItemVO], summary="创建空数据项")
async def create_dataset_item(
    body: DatasetItemCreateForm,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    result = await DatasetItemService.create_dataset_item(
        db, redis, body.model_dump(exclude_none=True),
    )
    return success(result, "创建成功")


@router.post("/upload", response_model=Result[DatasetItemVO], summary="创建数据项并上传配对图片")
async def upload_dataset_item_with_images(
    datasetId: int = Form(..., description="数据集ID"),
    name: Optional[str] = Form(default=None, description="数据项名称"),
    sceneType: Optional[str] = Form(default=None, description="场景类型"),
    clearImage: Optional[UploadFile] = File(default=None, description="清晰图文件（可选，适配无GT数据集）"),
    hazyImages: list[UploadFile] = File(default=[], description="有雾图文件列表（可选，适配仅有清晰图场景）"),
    hazeLevels: list[str] = Form(default=[], description="有雾图对应的雾霾程度列表，支持多种规范(light/medium/heavy/beta=0.5等)，可为空"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    levels = [lvl.strip() for lvl in hazeLevels if lvl.strip()]
    if len(levels) != len(hazyImages):
        raise BusinessException(ResultCode.PARAM_ERROR, "有雾图数量与雾霾程度数量不匹配")

    clear_content = await clearImage.read() if clearImage else None
    clear_ctype = clearImage.content_type if clearImage else ""

    hazy_data = []
    for i, hf in enumerate(hazyImages):
        content = await hf.read()
        ctype = hf.content_type or "application/octet-stream"
        hazy_data.append({
            "filename": hf.filename,
            "content": content,
            "contentType": ctype,
            "hazeLevel": levels[i] if i < len(levels) else "",
        })

    detail = await DatasetItemService.upload_dataset_item_with_images(
        db=db,
        redis=redis,
        dataset_id=datasetId,
        name=name,
        scene_type=sceneType,
        clear_file_content=clear_content,
        clear_filename=clearImage.filename or "" if clearImage else "",
        clear_content_type=clear_ctype,
        hazy_files_data=hazy_data,
    )
    return success(detail)


@router.post("/batch", response_model=Result[BatchUploadResultVO], summary="批量创建数据项并上传图片")
async def batch_create_dataset_items_with_images(
    datasetId: int = Form(..., description="数据集ID"),
    sceneType: Optional[str] = Form(default=None, description="场景类型"),
    files: list[UploadFile] = File(..., description="文件列表（混合清晰图+有雾图，按文件名自动配对）"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    files_data = []
    for f in files:
        content = await f.read()
        files_data.append({
            "filename": f.filename,
            "content": content,
            "contentType": f.content_type or "application/octet-stream",
        })

    result = await DatasetItemService.batch_create_dataset_items_with_images(
        db=db,
        redis=redis,
        dataset_id=datasetId,
        scene_type=sceneType,
        files_data=files_data,
    )
    return success(result)


@router.put("/{item_id}", response_model=Result[DatasetItemVO], summary="修改数据项信息")
async def update_dataset_item(
    item_id: int = Path(..., description="数据项ID"),
    body: DatasetItemUpdateForm = Body(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    result = await DatasetItemService.update_dataset_item(
        db, redis, item_id, body.model_dump(exclude_none=True),
    )
    return success(result, "更新成功")


@router.delete("/batch", response_model=Result[BatchOperationResultVO], summary="批量删除数据项")
async def batch_delete_dataset_items(
    body: BatchDeleteForm = Body(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    result = await DatasetItemService.batch_delete_items(db, redis, body.ids)
    return success(result, "删除成功")


@router.delete("/{item_id}", response_model=Result[None], summary="删除数据项")
async def delete_dataset_item(
    item_id: int = Path(..., description="数据项ID"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    await DatasetItemService.delete_dataset_item(db, redis, item_id)
    return success(msg="删除成功")
