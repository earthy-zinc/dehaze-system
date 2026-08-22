from fastapi import APIRouter, Body, Depends, Path, Query
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import success
from app.database import get_db
from app.decorators import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.feedback import (
    FeedbackAssignForm,
    FeedbackCloseForm,
    FeedbackCreateForm,
    FeedbackReplyForm,
    FeedbackSupplementForm,
    RatingCreateForm,
    RatingReplyForm,
)
from app.service.feedback_service import feedback_service

router = APIRouter(
    prefix="/api/v1/feedback",
    tags=["反馈评价"],
    dependencies=[Depends(get_current_user)],
)


# ============ 评价接口 ============


@router.post("/ratings", summary="提交评分")
async def create_rating(
    body: RatingCreateForm = Body(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    data = await feedback_service.create_rating(
        db, redis, user.id, body.model_dump(exclude_none=True)
    )
    return success(data)


@router.get("/ratings/my", summary="我的评价列表")
async def list_my_ratings(
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await feedback_service.list_my_ratings(
        db, user.id, {"pageNum": pageNum, "pageSize": pageSize}
    )
    return success(data)


@router.get("/ratings/by-prediction/{prediction_log_id}", summary="按处理记录查评价")
async def get_rating_by_prediction(
    prediction_log_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await feedback_service.get_rating_by_prediction(db, user.id, prediction_log_id)
    return success(data)


@router.get("/ratings/page", summary="评价分页列表")
async def list_ratings(
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    keywords: str | None = Query(default=None),
    algorithmId: int | None = Query(default=None),
    ratingMin: int | None = Query(default=None, ge=1, le=5),
    ratingMax: int | None = Query(default=None, ge=1, le=5),
    hasComment: bool | None = Query(default=None),
    startTime: str | None = Query(default=None),
    endTime: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await feedback_service.list_paged_ratings(
        db,
        {
            "pageNum": pageNum,
            "pageSize": pageSize,
            "keywords": keywords,
            "algorithmId": algorithmId,
            "ratingMin": ratingMin,
            "ratingMax": ratingMax,
            "hasComment": hasComment,
            "startTime": startTime,
            "endTime": endTime,
        },
    )
    return success(data)


@router.get("/ratings/stats", summary="评价统计")
async def get_rating_stats(
    startTime: str | None = Query(default=None),
    endTime: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    data = await feedback_service.get_rating_stats(db, redis, startTime, endTime)
    return success(data)


@router.put("/ratings/{rating_id}", summary="修改评分")
async def update_rating(
    rating_id: int = Path(...),
    body: RatingCreateForm = Body(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await feedback_service.update_rating(
        db, redis, user.id, rating_id, body.model_dump(exclude_none=True)
    )
    return success()


@router.put("/ratings/{rating_id}/hide", summary="隐藏评价")
@require_permission("feedback:rating:edit")
async def hide_rating(
    rating_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await feedback_service.hide_rating(db, rating_id)
    return success()


@router.post("/ratings/{rating_id}/reply", summary="回复评价")
@require_permission("feedback:rating:reply")
async def reply_rating(
    rating_id: int = Path(...),
    body: RatingReplyForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await feedback_service.reply_rating(db, rating_id, body.content, user.id)
    return success()


# ============ 反馈接口 ============


@router.post("", summary="提交反馈")
async def create_feedback(
    body: FeedbackCreateForm = Body(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    data = await feedback_service.create_feedback(
        db, redis, user.id, body.model_dump(exclude_none=True)
    )
    return success(data)


@router.get("/my", summary="我的反馈列表")
async def list_my_feedback(
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await feedback_service.list_my_feedback(
        db, user.id, {"pageNum": pageNum, "pageSize": pageSize}
    )
    return success(data)


@router.get("/page", summary="反馈分页列表")
async def list_feedback(
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    keywords: str | None = Query(default=None),
    feedbackType: str | None = Query(default=None),
    status: str | None = Query(default=None),
    relatedModule: str | None = Query(default=None),
    priority: int | None = Query(default=None),
    assigneeId: int | None = Query(default=None),
    startTime: str | None = Query(default=None),
    endTime: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await feedback_service.list_paged_feedback(
        db,
        {
            "pageNum": pageNum,
            "pageSize": pageSize,
            "keywords": keywords,
            "feedbackType": feedbackType,
            "status": status,
            "relatedModule": relatedModule,
            "priority": priority,
            "assigneeId": assigneeId,
            "startTime": startTime,
            "endTime": endTime,
        },
    )
    return success(data)


@router.get("/stats", summary="反馈统计")
async def get_feedback_stats(
    startTime: str | None = Query(default=None),
    endTime: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    data = await feedback_service.get_feedback_stats(db, redis, startTime, endTime)
    return success(data)


@router.get("/{feedback_id}", summary="反馈详情")
async def get_feedback_detail(
    feedback_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await feedback_service.get_feedback_detail(db, feedback_id, user.id, user.is_admin)
    return success(data)


@router.post("/{feedback_id}/supplement", summary="补充说明")
async def supplement_feedback(
    feedback_id: int = Path(...),
    body: FeedbackSupplementForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await feedback_service.supplement_feedback(
        db, user.id, feedback_id, body.model_dump(exclude_none=True)
    )
    return success()


@router.put("/{feedback_id}/assign", summary="分配处理人")
@require_permission("feedback:assign")
async def assign_feedback(
    feedback_id: int = Path(...),
    body: FeedbackAssignForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await feedback_service.assign_feedback(db, feedback_id, body.assigneeId, user.id)
    return success()


@router.post("/{feedback_id}/reply", summary="回复反馈")
@require_permission("feedback:reply")
async def reply_feedback(
    feedback_id: int = Path(...),
    body: FeedbackReplyForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await feedback_service.reply_feedback(
        db, feedback_id, body.model_dump(exclude_none=True), user.id
    )
    return success()


@router.put("/{feedback_id}/close", summary="关闭反馈")
@require_permission("feedback:close")
async def close_feedback(
    feedback_id: int = Path(...),
    body: FeedbackCloseForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await feedback_service.close_feedback(db, feedback_id, body.closeReason, user.id)
    return success()


@router.put("/{feedback_id}/tags", summary="设置反馈标签")
@require_permission("feedback:edit")
async def update_feedback_tags(
    feedback_id: int = Path(...),
    tags: list[str] = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await feedback_service.update_feedback_tags(db, feedback_id, tags)
    return success()
