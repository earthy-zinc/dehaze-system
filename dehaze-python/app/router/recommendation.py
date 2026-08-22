"""
推荐管理路由
"""

from fastapi import APIRouter, Body, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import success
from app.database import get_db
from app.decorators.permission import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.recommendation import (
    AnalyzeForm,
    RecommendationFeedbackForm,
    RecommendationRuleForm,
)
from app.service.recommendation_service import RecommendationService

router = APIRouter(
    prefix="/api/v1/recommendations",
    tags=["推荐管理"],
    dependencies=[Depends(get_current_user)],
)


@router.post("/analyze", summary="图像特征分析")
async def analyze(
    body: AnalyzeForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await RecommendationService.analyze(body.imageId, body.imageUrl)
    return success(data.model_dump())


@router.get("/algorithms", summary="获取算法推荐")
async def get_algorithms(
    analysisId: int | None = Query(default=None),
    imageMd5: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await RecommendationService.get_algorithms(db, user.id, analysisId, imageMd5)
    return success([item.model_dump() for item in data])


@router.post("/feedback", summary="推荐反馈")
async def submit_feedback(
    body: RecommendationFeedbackForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await RecommendationService.submit_feedback(db, body.recommendationId, body.useful)
    return success(data.model_dump())


@router.get("/rules", summary="获取推荐规则")
@require_permission("sys:recommendation:rule:view")
async def get_rules(
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await RecommendationService.get_rules(db)
    return success([item.model_dump() for item in data])


@router.put("/rules", summary="更新/新增推荐规则")
@require_permission("sys:recommendation:rule:edit")
async def update_rule(
    body: RecommendationRuleForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    rule_id = body.id if body.id else 0
    data = await RecommendationService.update_rule(db, rule_id, body.model_dump())
    return success(data.id)


@router.get("/report", summary="推荐效果报表")
@require_permission("sys:recommendation:report")
async def get_report(
    startDate: str | None = Query(default=None),
    endDate: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await RecommendationService.get_report(db, startDate, endDate)
    return success(data.model_dump())
