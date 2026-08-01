"""
算法选择 API 路由

基础路径: /api/v1/algorithms/select
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.algorithm_select import (
    AlgorithmCompareVO,
    AlgorithmDetailVO,
    AlgorithmSearchVO,
    AlgorithmTreeNodeVO,
    CompareRequest,
    TestRequest,
    TestResultVO,
)
from app.service.algorithm_select_service import AlgorithmSelectService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/algorithms/select",
    tags=["算法选择"],
    dependencies=[Depends(get_current_user)],
)


@router.get(
    "/tree",
    response_model=Result[list[AlgorithmTreeNodeVO]],
    summary="算法选择树",
)
async def get_algorithm_tree(
    db: AsyncSession = Depends(get_db),
):
    """获取算法选择树（仅返回已发布状态的算法）"""
    tree = await AlgorithmSelectService.get_algorithm_tree(db)
    return success(tree)


@router.get(
    "/search",
    response_model=Result[list[AlgorithmSearchVO]],
    summary="搜索算法",
)
async def search_algorithms(
    keyword: Optional[str] = Query(default=None, description="搜索关键词"),
    db: AsyncSession = Depends(get_db),
):
    """搜索算法（关键词/拼音/标签）"""
    results = await AlgorithmSelectService.search_algorithms(db, keyword=keyword)
    return success(results)


@router.get(
    "/{algorithm_id}",
    response_model=Result[AlgorithmDetailVO],
    summary="算法详情",
)
async def get_algorithm_detail(
    algorithm_id: int,
    db: AsyncSession = Depends(get_db),
):
    """获取算法详情（含样例效果图、评分、使用次数）"""
    detail = await AlgorithmSelectService.get_algorithm_detail(db, algorithm_id)
    return success(detail)


@router.post(
    "/{algorithm_id}/test",
    response_model=Result[TestResultVO],
    summary="测试算法效果",
)
async def test_algorithm(
    algorithm_id: int,
    body: TestRequest,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """上传自定义图片测试算法效果"""
    result = await AlgorithmSelectService.test_algorithm(
        db=db,
        algorithm_id=algorithm_id,
        image_url=body.imageUrl,
        user_id=user.id,
    )
    return success(result)


@router.post(
    "/compare",
    response_model=Result[list[AlgorithmCompareVO]],
    summary="算法对比",
)
async def compare_algorithms(
    body: CompareRequest,
    db: AsyncSession = Depends(get_db),
):
    """算法对比（最多3个算法）"""
    result = await AlgorithmSelectService.compare(
        db=db,
        algorithm_ids=body.algorithmIds,
    )
    return success(result)
