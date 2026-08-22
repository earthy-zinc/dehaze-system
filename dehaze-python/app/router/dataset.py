"""
数据集路由

基础路径: /api/v1/datasets
"""

from fastapi import APIRouter, Body, Depends, Path, Query
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import success
from app.database import get_db
from app.decorators.permission import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.common import BatchDeleteForm
from app.models.schema.dataset import DatasetAddForm, DatasetUpdateForm
from app.service.dataset.dataset_service import dataset_service

router = APIRouter(
    prefix="/api/v1/datasets",
    tags=["数据集管理"],
    dependencies=[Depends(get_current_user)],
)


@router.get("", summary="分页查询数据集列表")
async def list_datasets(
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=10, ge=1, le=100, description="每页数量"),
    keyword: str | None = Query(default=None, description="关键词(数据集名称)"),
    type: str | None = Query(default=None, description="数据集类型"),
    status: int | None = Query(default=None, description="状态(1:启用；0:禁用)"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    result = await dataset_service.get_page(db, redis, pageNum, pageSize, keyword, type, status)
    return success(result)


@router.get("/children/{parent_id}", summary="获取子数据集列表（懒加载）")
async def list_children(
    parent_id: int = Path(..., description="父数据集ID"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    children = await dataset_service.get_children(db, redis, parent_id)
    return success(children)


@router.get("/options", summary="获取数据集下拉选项")
async def list_dataset_options(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    options = await dataset_service.get_dataset_options(db, redis)
    return success(options)


@router.get("/evaluation-options", summary="测试集选项查询（评估接入）")
async def list_evaluation_options(
    taskType: str | None = Query(default=None, description="任务类型"),
    db: AsyncSession = Depends(get_db),
):
    """测试集选项查询：按 taskType 过滤，仅返回含清晰图 GT 的数据集（T-DS-046~048）。

    注意：必须声明在 /{dataset_id} 之前，避免被路径参数路由吞掉。
    """
    options = await dataset_service.get_evaluation_options(db, task_type=taskType)
    return success(options)


@router.get("/{dataset_id}", summary="获取数据集详情")
async def get_dataset(
    dataset_id: int = Path(..., description="数据集ID"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    dataset = await dataset_service.get_dataset_by_id(db, redis, dataset_id)
    return success(dataset)


@router.post("", summary="新增数据集")
@require_permission("sys:dataset:add")
async def create_dataset(
    body: DatasetAddForm,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await dataset_service.create_dataset(db, redis, body.model_dump(exclude_none=True))
    return success(result)


@router.put("/{dataset_id}", summary="修改数据集")
@require_permission("sys:dataset:edit")
async def update_dataset(
    dataset_id: int = Path(..., description="数据集ID"),
    body: DatasetUpdateForm = Body(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await dataset_service.update_dataset(
        db, redis, dataset_id, body.model_dump(exclude_none=True)
    )
    return success(result)


@router.delete("/batch", summary="批量删除数据集")
@require_permission("sys:dataset:delete")
async def batch_delete_datasets(
    body: BatchDeleteForm = Body(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await dataset_service.delete_datasets(db, redis, body.ids)
    return success(result)


@router.delete("/{dataset_id}", summary="删除单个数据集")
@require_permission("sys:dataset:delete")
async def delete_dataset(
    dataset_id: int = Path(..., description="数据集ID"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await dataset_service.delete_dataset(db, redis, dataset_id)
    return success()
