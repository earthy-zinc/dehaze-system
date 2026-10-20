from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.message import NotificationSettingsForm
from app.service.notification_setting_service import notification_setting_service

router = APIRouter(
    prefix="/api/v1/notification-settings",
    tags=["通知设置"],
    dependencies=[Depends(get_current_user)],
)


@router.get("", summary="获取通知偏好设置")
async def get_notification_settings(
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await notification_setting_service.get_or_init(db, user.id)
    return success(data)


@router.patch("", summary="更新通知偏好设置")
async def update_notification_settings(
    body: NotificationSettingsForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await notification_setting_service.update(db, user.id, body.model_dump(exclude_none=True))
    return success()
