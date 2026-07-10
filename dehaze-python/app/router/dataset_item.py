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
from app.models.schema.dataset import (BatchUploadResultVO,
                                       DatasetItemCreateForm, DatasetItemIdVO,
                                       DatasetItemPageVO,
                                       DatasetItemUpdateForm, DatasetItemVO)
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
    datasetId: int = Query(..., description="所属数据集ID"),
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=20, ge=1, le=100, description="每页数量"),
    keywords: Optional[str] = Query(default=None, description="搜索关键词"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    result = await DatasetService.get_image_items(
        db, redis, datasetId, pageNum, pageSize, keywords,
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


@router.post("", response_model=Result[DatasetItemIdVO], summary="创建空数据项")
async def create_dataset_item(
    body: DatasetItemCreateForm,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    item_id = await DatasetItemService.create_dataset_item(
        db, redis, body.model_dump(exclude_none=True),
    )
    return success(DatasetItemIdVO(id=item_id), "创建成功")


@router.post("/upload", response_model=Result[DatasetItemVO], summary="创建数据项并上传配对图片")
async def upload_dataset_item_with_images(
    datasetId: int = Form(..., description="数据集ID"),
    name: Optional[str] = Form(default=None, description="数据项名称"),
    sceneType: Optional[str] = Form(default=None, description="场景类型"),
    clearFile: UploadFile = File(..., description="清晰图文件"),
    hazyFiles: list[UploadFile] = File(..., description="有雾图文件列表"),
    hazeLevels: str = Form(..., description="雾霾程度列表，逗号分隔(light/medium/heavy)"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    # 解析雾霾程度
    levels = [lvl.strip() for lvl in hazeLevels.split(",") if lvl.strip()]
    if len(levels) != len(hazyFiles):
        raise BusinessException(ResultCode.PARAM_ERROR, "有雾图数量与雾霾程度数量不匹配")

    # 读取文件内容
    clear_content = await clearFile.read()
    clear_ctype = clearFile.content_type or "application/octet-stream"

    hazy_data = []
    for i, hf in enumerate(hazyFiles):
        content = await hf.read()
        ctype = hf.content_type or "application/octet-stream"
        hazy_data.append({
            "filename": hf.filename,
            "content": content,
            "contentType": ctype,
            "hazeLevel": levels[i] if i < len(levels) else "medium",
        })

    detail = await DatasetItemService.upload_dataset_item_with_images(
        db=db,
        redis=redis,
        dataset_id=datasetId,
        name=name,
        scene_type=sceneType,
        clear_file_content=clear_content,
        clear_filename=clearFile.filename or "",
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


@router.put("/{item_id}", response_model=Result[None], summary="修改数据项信息")
async def update_dataset_item(
    item_id: int = Path(..., description="数据项ID"),
    body: DatasetItemUpdateForm = Body(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    await DatasetItemService.update_dataset_item(
        db, redis, item_id, body.model_dump(exclude_none=True),
    )
    return success(msg="更新成功")


@router.delete("/{item_id}", response_model=Result[None], summary="删除数据项")
async def delete_dataset_item(
    item_id: int = Path(..., description="数据项ID"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    await DatasetItemService.delete_dataset_item(db, redis, item_id)
    return success(msg="删除成功")


@router.delete("/batch", response_model=Result[None], summary="批量删除数据项")
async def batch_delete_dataset_items(
    body: BatchDeleteForm = Body(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    await DatasetItemService.batch_delete_items(db, redis, body.ids)
    return success(msg="删除成功")
