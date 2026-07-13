"""
图像输入历史记录 API 路由

GET    /api/v1/image-input/history         → 分页查询历史记录
POST   /api/v1/image-input/history          → 创建历史记录
POST   /api/v1/image-input/history/sync     → 同步本地与云端
DELETE /api/v1/image-input/history/batch    → 批量删除
DELETE /api/v1/image-input/history/clear    → 清空历史
GET    /api/v1/image-input/history/{id}     → 历史记录详情
PUT    /api/v1/image-input/history/{id}     → 更新（如收藏）
DELETE /api/v1/image-input/history/{id}     → 删除单条

注意：静态路径（/batch, /clear, /sync）必须注册在动态路径 /{id} 之前，
否则 FastAPI 会将 /batch 匹配到 /{id} 路由，导致 int 转换失败返回 400。
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.common import PageResult
from app.models.schema.input_history import (
    InputHistoryForm,
    InputHistoryUpdateForm,
    InputHistoryVO,
)
from app.service.input_history_service import InputHistoryService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/image-input/history",
    tags=["图像输入历史记录"],
)


# ── 静态路径（必须 before /{id}）──────────────────────────


@router.get("", response_model=Result[PageResult[InputHistoryVO]], summary="分页查询历史记录")
async def list_history(
    status: Optional[int] = Query(default=None, description="状态筛选（1=成功，2=失败，3=处理中）"),
    inputSource: Optional[str] = Query(default=None, description="图片来源筛选: upload/camera/sample"),
    isFavorite: Optional[bool] = Query(default=None, description="仅收藏"),
    keywords: Optional[str] = Query(default=None, description="关键词"),
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=10, ge=1, le=100, description="每页数量"),
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    分页查询当前用户的历史记录（按用户隔离）

    游客无法使用此接口（需登录后才能云端存储历史）
    """
    list_vo, total = await InputHistoryService.list_history(
        db=db,
        user_id=user.id,
        status=status,
        input_source=inputSource,
        favorite_only=isFavorite == True,
        keywords=keywords,
        page=pageNum,
        size=pageSize,
    )
    return success(PageResult(list=list_vo, total=total))


@router.post("", response_model=Result[int], summary="创建历史记录")
async def create_history(
    body: InputHistoryForm,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """创建历史记录（关联当前用户）"""
    history_id = await InputHistoryService.create_history(
        db=db,
        data=body.model_dump(by_alias=False, exclude_none=True),
        user_id=user.id,
    )
    return success(history_id, msg="创建成功")


@router.post("/sync", response_model=Result[int], summary="同步本地与云端")
async def sync_history(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """同步历史记录（标记所有未同步记录为已同步）"""
    result = await InputHistoryService.sync_history(db, user.id)
    return success(result)


@router.delete("/batch", response_model=Result[int], summary="批量删除历史记录")
async def batch_delete_history(
    body: dict = Body(...),
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """批量删除历史记录（仅限本人）"""
    ids = body.get("ids", [])
    count = await InputHistoryService.batch_delete(db, ids, user_id=user.id)
    return success(count)


@router.delete("/clear", response_model=Result[int], summary="清空历史记录")
async def clear_history(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """清空当前用户的所有历史记录"""
    count = await InputHistoryService.clear_history(db, user.id)
    return success(count)


# ── 动态路径 /{id}（必须 after 静态路径）─────────────────


@router.get("/{history_id}", response_model=Result[InputHistoryVO], summary="历史记录详情")
async def get_history(
    history_id: int,
    db: AsyncSession = Depends(get_db),
):
    """查询历史记录详情"""
    history = await InputHistoryService.get_history(db, history_id)
    if not history:
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "历史记录不存在")
    return success(history)


@router.put("/{history_id}", response_model=Result[None], summary="更新历史记录")
async def update_history(
    history_id: int,
    body: InputHistoryUpdateForm,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    更新历史记录（如添加收藏、补充处理结果）

    仅本人可操作自己的历史记录
    """
    await InputHistoryService.update_history(
        db=db,
        history_id=history_id,
        is_favorite=body.isFavorite,
        user_id=user.id,
    )
    return success(msg="更新成功")


@router.delete("/{history_id}", response_model=Result[None], summary="删除单条历史记录")
async def delete_history(
    history_id: int,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """删除单条历史记录（仅限本人，幂等）"""
    await InputHistoryService.delete_history(db, history_id, user_id=user.id)
    return success(msg="删除成功")
