"""
算法选择 API 路由

POST /api/v1/algorithm-select/recommend   → 智能推荐
POST /api/v1/algorithm-select/favorite     → 收藏/取消
GET  /api/v1/algorithm-select/favorites    → 收藏列表
POST /api/v1/algorithm-select/compare      → 算法对比
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success, error
from app.core.code import ResultCode
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.algorithm_select import (
    AlgorithmCompareVO,
    AlgorithmRecommendVO,
    CompareRequest,
    FavoriteForm,
    FavoriteVO,
    RecommendRequest,
)
from app.service.algorithm_select_service import AlgorithmSelectService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/algorithm-select",
    tags=["算法选择"],
    dependencies=[Depends(get_current_user)],
)


@router.post(
    "/recommend",
    response_model=Result[list[AlgorithmRecommendVO]],
    summary="智能推荐算法",
)
async def recommend_algorithms(
    body: RecommendRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    基于图像特征分析返回 Top N 算法推荐

    特征权重:
    - 雾霾浓度 30%
    - 场景类型 20%
    - 光照 15%
    - 复杂度 10%
    - 颜色 10%
    - 分辨率 5%
    - 用户偏好 10%
    """
    recommendations = await AlgorithmSelectService.recommend(
        db=db,
        image_url=body.imageUrl,
        top_n=body.topN,
    )
    return success(recommendations)


@router.post(
    "/favorite",
    response_model=Result[dict],
    summary="收藏/取消收藏算法",
)
async def toggle_favorite(
    body: FavoriteForm,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    切换收藏状态

    - 未收藏 → 添加收藏
    - 已收藏 → 取消收藏
    """
    # 检查是否已收藏
    from app.service.algorithm_select_service import AlgorithmSelectService
    existing = await AlgorithmSelectService._get_favorite(db, user.id, body.algorithmId)
    if existing:
        # 已收藏，取消
        await AlgorithmSelectService.remove_favorite(db, user.id, body.algorithmId)
        return success({"favorited": False}, msg="已取消收藏")
    else:
        # 未收藏，添加
        fav_id = await AlgorithmSelectService.add_favorite(db, user.id, body.algorithmId)
        return success({"favorited": True, "favoriteId": fav_id}, msg="收藏成功")


@router.get(
    "/favorites",
    response_model=Result[list[FavoriteVO]],
    summary="收藏列表",
)
async def list_favorites(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """查询当前用户的算法收藏列表"""
    favorites = await AlgorithmSelectService.list_favorites(db, user.id)
    return success(favorites)


@router.post(
    "/compare",
    response_model=Result[list[AlgorithmCompareVO]],
    summary="算法对比",
)
async def compare_algorithms(
    body: CompareRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    多算法对比（最多 4 个）

    返回多个算法的元数据对比。
    实际去雾效果对比需前端分别调用 /prediction 接口获取结果。
    """
    result = await AlgorithmSelectService.compare(
        db=db,
        algorithm_ids=body.algorithmIds,
        image_url=body.imageUrl,
    )
    return success(result)
