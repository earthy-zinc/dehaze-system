"""
收藏管理路由

基础路径：/api/v1/favorites
"""

from typing import Optional

from fastapi import APIRouter, Body, Depends, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.favorite import FavoriteCreateForm
from app.service.favorite_service import FavoriteService

router = APIRouter(
    prefix="/api/v1/favorites",
    tags=["收藏管理"],
    dependencies=[Depends(get_current_user)],
)


@router.get("/page", summary="收藏列表分页查询")
async def get_page(
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=20, ge=1, le=100),
    targetType: Optional[str] = Query(default=None),
    keywords: Optional[str] = Query(default=None),
    sortBy: Optional[str] = Query(default="create_time"),
    sortOrder: Optional[str] = Query(default="desc"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await FavoriteService.get_page(
        db,
        user.id,
        {
            "pageNum": pageNum,
            "pageSize": pageSize,
            "targetType": targetType,
            "keywords": keywords,
            "sortBy": sortBy,
            "sortOrder": sortOrder,
        },
    )
    return success(data)


@router.post("", summary="添加收藏")
async def add(
    body: FavoriteCreateForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    favorite_id = await FavoriteService.add(
        db, user.id, body.targetType, body.targetId
    )
    return success(favorite_id)


@router.delete("/{ids}", summary="批量取消收藏")
async def delete_by_ids(
    ids: str = Path(..., description="收藏记录ID列表，逗号分隔"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    id_list = [int(x) for x in ids.split(",") if x.strip().isdigit()]
    await FavoriteService.delete_by_ids(db, user.id, id_list)
    return success()


@router.get("/{target_id}/status", summary="检查是否已收藏")
async def get_status(
    target_id: int = Path(..., description="收藏对象ID"),
    targetType: str = Query(..., description="收藏对象类型"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await FavoriteService.get_status(db, user.id, targetType, target_id)
    return success(data)


@router.get("/count", summary="收藏数量统计")
async def get_count(
    targetType: Optional[str] = Query(default=None, description="按类型筛选"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await FavoriteService.get_count(db, user.id, targetType)
    return success(data)
