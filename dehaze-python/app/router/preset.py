"""参数预设 API 路由"""

import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.common import PageResult
from app.models.schema.preset import PresetForm, PresetVO
from app.service.preset_service import preset_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/presets", tags=["参数预设"], dependencies=[Depends(get_current_user)]
)


@router.get("", response_model=Result[PageResult[PresetVO]], summary="参数预设列表")
async def list_presets(
    algorithmId: int | None = Query(default=None, description="算法ID（可选筛选）"),
    isSystem: bool | None = Query(default=None, description="是否系统预设"),
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=10, ge=1, le=100, description="每页数量"),
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """获取参数预设列表（系统预设 + 用户自定义）"""
    result = await preset_service.list_presets(
        db,
        user.id,
        algorithmId,
        is_system=isSystem,
        page=pageNum,
        size=pageSize,
    )
    return success(result)


@router.post("", response_model=Result[PresetVO], summary="创建自定义预设")
async def create_preset(
    form: PresetForm,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """创建自定义预设"""
    result = await preset_service.create_preset(db, user.id, form)
    return success(result)


@router.put("/{id}", response_model=Result[PresetVO], summary="更新自定义预设")
async def update_preset(
    id: int,
    form: PresetForm,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """更新自定义预设"""
    result = await preset_service.update_preset(db, id, user.id, form)
    return success(result)


@router.delete("/{id}", response_model=Result[None], summary="删除自定义预设")
async def delete_preset(
    id: int,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """删除自定义预设"""
    await preset_service.delete_preset(db, id, user.id)
    return success()
