"""用户 AI 偏好：偏好默认对话模型的读写（登录用户本人，无权限码）。"""

from fastapi import APIRouter, Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.ai_user_preference import (
    AiUserPreferenceResult,
    AiUserPreferenceUpdate,
)
from app.service.ai_user_preference_service import ai_user_preference_service

router = APIRouter(prefix="/api/v1/ai", tags=["AI用户偏好"])


@router.get(
    "/user-preference",
    response_model=Result[AiUserPreferenceResult],
    summary="获取我的默认模型偏好",
)
async def get_user_preference(
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    """偏好默认对话模型（`defaultModel` 为 null 表示未设置，跟随平台默认模型）"""
    model_id = await ai_user_preference_service.get_default_model(db, user.id)
    return success(AiUserPreferenceResult(default_model=model_id))


@router.put(
    "/user-preference",
    response_model=Result[None],
    summary="设置我的默认模型偏好",
)
async def update_user_preference(
    form: AiUserPreferenceUpdate,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    """全量替换本人偏好：`defaultModel` 为 null 或缺省即清除（回落平台默认模型），仅影响新建会话"""
    await ai_user_preference_service.set_default_model(db, redis, user.id, form.default_model)
    return success(msg="一切ok")
