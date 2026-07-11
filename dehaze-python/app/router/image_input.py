"""
图像输入历史记录 API 路由

GET    /api/v1/image-input/history         → 分页查询历史记录
GET    /api/v1/image-input/history/{id}    → 历史记录详情
POST   /api/v1/image-input/history          → 创建历史记录
PUT    /api/v1/image-input/history/{id}    → 更新（如收藏）
DELETE /api/v1/image-input/history/{id}    → 删除单条
DELETE /api/v1/image-input/history/batch   → 批量删除
DELETE /api/v1/image-input/history/clear   → 清空历史
POST   /api/v1/image-input/history/sync    → 同步本地与云端
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success, error
from app.core.code import ResultCode
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.common import PageResult
from app.models.schema.input_history import (
    BatchDeleteForm,
    InputHistoryForm,
    InputHistoryUpdateForm,
    InputHistoryVO,
    SyncResultVO,
)
from app.service.input_history_service import InputHistoryService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/image-input/history",
    tags=["图像输入历史记录"],
)


@router.get("", response_model=Result[PageResult[InputHistoryVO]], summary="分页查询历史记录")
async def list_history(
    inputSource: Optional[str] = Query(default=None, description="图片来源筛选: upload/camera/sample"),
    favoriteOnly: bool = Query(default=False, description="仅收藏"),
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
        input_source=inputSource,
        favorite_only=favoriteOnly,
        keywords=keywords,
        page=pageNum,
        size=pageSize,
    )
    return success(PageResult(list=list_vo, total=total))


@router.get("/{history_id}", response_model=Result[InputHistoryVO], summary="历史记录详情")
async def get_history(
    history_id: int,
    db: AsyncSession = Depends(get_db),
):
    """查询历史记录详情"""
    history = await InputHistoryService.get_history(db, history_id)
    if not history:
        return error("历史记录不存在", ResultCode.RESOURCE_NOT_FOUND.code)
    return success(history)


@router.post("", response_model=Result[InputHistoryVO], summary="创建历史记录")
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
    history = await InputHistoryService.get_history(db, history_id)
    return success(history, msg="历史记录创建成功")


@router.put("/{history_id}", response_model=Result[InputHistoryVO], summary="更新历史记录")
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
        data=body.model_dump(by_alias=False, exclude_none=True),
        user_id=user.id,
    )
    history = await InputHistoryService.get_history(db, history_id)
    return success(history, msg="历史记录更新成功")


@router.delete("/{history_id}", response_model=Result[None], summary="删除单条历史记录")
async def delete_history(
    history_id: int,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """删除单条历史记录（仅限本人）"""
    await InputHistoryService.delete_history(db, history_id, user_id=user.id)
    return success(msg="历史记录已删除")


@router.delete("/batch", response_model=Result[dict], summary="批量删除历史记录")
async def batch_delete_history(
    body: BatchDeleteForm,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """批量删除历史记录（仅限本人）"""
    count = await InputHistoryService.batch_delete(db, body.ids, user_id=user.id)
    return success({"count": count}, msg=f"已删除 {count} 条历史记录")


@router.delete("/clear", response_model=Result[dict], summary="清空历史记录")
async def clear_history(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """清空当前用户的所有历史记录"""
    count = await InputHistoryService.clear_history(db, user.id)
    return success({"count": count}, msg=f"已清空 {count} 条历史记录")


@router.post("/sync", response_model=Result[SyncResultVO], summary="同步本地与云端")
async def sync_history(
    items: List[dict],
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    同步本地历史记录到云端

    将本地存储的历史记录批量上传到云端，支持去重
    """
    result = await InputHistoryService.sync_history(db, items, user.id)
    return success(SyncResultVO(
        synced=result["synced"],
        failed=result["failed"],
        message=f"同步完成: 成功 {result['synced']} 条, 失败 {result['failed']} 条",
    ))
