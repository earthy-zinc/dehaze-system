"""
数据集路由

基础路径: /api/v1/datasets
"""
from typing import Optional

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.dataset import (DatasetAddForm, DatasetDeleteResultVO,
                                       DatasetIdVO, DatasetOptionVO,
                                       DatasetUpdateForm, DatasetVO)
from app.service.dataset_service import DatasetService
from fastapi import APIRouter, Body, Depends, Path, Query
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(
    prefix="/api/v1/datasets",
    tags=["数据集管理"],
    dependencies=[Depends(get_current_user)],
)


@router.get("", response_model=Result[list[DatasetVO]], summary="获取数据集列表")
async def list_datasets(
    keywords: Optional[str] = Query(default=None, description="关键词(数据集名称)"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    datasets = await DatasetService.get_dataset_tree(db, redis, keywords)
    return success(datasets)


@router.get("/options", response_model=Result[list[DatasetOptionVO]], summary="获取数据集下拉选项")
async def list_dataset_options(
    db: AsyncSession = Depends(get_db),
):
    options = await DatasetService.get_dataset_options(db)
    return success(options)


@router.get("/{dataset_id}", response_model=Result[DatasetVO], summary="获取数据集详情")
async def get_dataset(
    dataset_id: int = Path(..., description="数据集ID"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    dataset = await DatasetService.get_dataset_by_id(db, redis, dataset_id)
    if not dataset:
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据集不存在")
    return success(dataset)


@router.post("", response_model=Result[DatasetIdVO], summary="新增数据集")
async def create_dataset(
    body: DatasetAddForm,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    dataset_id = await DatasetService.create_dataset(db, redis, body.model_dump(exclude_none=True))
    return success(DatasetIdVO(id=dataset_id), "创建成功")


@router.put("/{dataset_id}", response_model=Result[DatasetIdVO], summary="修改数据集")
async def update_dataset(
    dataset_id: int = Path(..., description="数据集ID"),
    body: DatasetUpdateForm = Body(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    await DatasetService.update_dataset(db, redis, dataset_id, body.model_dump(exclude_none=True))
    return success(DatasetIdVO(id=dataset_id), "更新成功")


@router.delete("/{dataset_id}", response_model=Result[DatasetDeleteResultVO], summary="删除单个数据集")
async def delete_dataset(
    dataset_id: int = Path(..., description="数据集ID"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    result = await DatasetService.delete_datasets(db, redis, [dataset_id])
    return success(result["data"], result["message"])


@router.delete("/batch", response_model=Result[DatasetDeleteResultVO], summary="批量删除数据集")
async def batch_delete_datasets(
    ids: str = Query(..., description="数据集ID列表，多个以英文逗号(,)分隔"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    if not ids:
        raise BusinessException(ResultCode.PARAM_ERROR, "参数错误")

    try:
        dataset_ids = [int(id_str.strip())
                       for id_str in ids.split(",") if id_str.strip()]
    except ValueError:
        raise BusinessException(ResultCode.PARAM_ERROR, "参数格式错误，ID 必须为数字")

    result = await DatasetService.delete_datasets(db, redis, dataset_ids)
    return success(result["data"], result["message"])
