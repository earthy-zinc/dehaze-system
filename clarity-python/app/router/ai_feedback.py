from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.ai_feedback import FeedbackCreateRequest, FeedbackResult
from app.service.ai_feedback_service import ai_feedback_service

router = APIRouter(prefix="/api/v1/ai", tags=["AI对话"])


@router.post(
    "/messages/{message_id}/feedback",
    response_model=Result[FeedbackResult],
    summary="提交/更新消息反馈",
)
async def submit_feedback(
    message_id: int,
    form: FeedbackCreateRequest,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_feedback_service.submit_feedback(db, message_id, user.id, form)
    return success(result)


@router.get(
    "/messages/{message_id}/feedback",
    response_model=Result[FeedbackResult | None],
    summary="查询消息反馈状态",
)
async def get_feedback(
    message_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_feedback_service.get_feedback(db, message_id, user.id)
    return success(result)


@router.delete(
    "/messages/{message_id}/feedback", response_model=Result[None], summary="撤销消息反馈"
)
async def revoke_feedback(
    message_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await ai_feedback_service.revoke_feedback(db, message_id, user.id)
    return success(msg="一切ok")
