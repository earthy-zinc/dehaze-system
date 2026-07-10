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
from app.models.schema.dataset import (DatasetItemCreateForm, DatasetItemIdVO,
                                       DatasetItemPageVO,
                                       DatasetItemUpdateForm, DatasetItemVO)
from app.service.dataset_service import DatasetItemService, DatasetService
from fastapi import APIRouter, Body, Depends, Path, Query
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
    ids: str = Query(..., description="数据项ID列表，多个以英文逗号(,)分隔"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    if not ids:
        raise BusinessException(ResultCode.PARAM_ERROR, "参数错误")

    try:
        item_ids = [int(id_str.strip())
                    for id_str in ids.split(",") if id_str.strip()]
    except ValueError:
        raise BusinessException(ResultCode.PARAM_ERROR, "参数格式错误，ID 必须为数字")

    await DatasetItemService.batch_delete_items(db, redis, item_ids)
    return success(msg="删除成功")
