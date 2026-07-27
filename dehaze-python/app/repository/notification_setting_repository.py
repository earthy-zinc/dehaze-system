from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_notification_setting import SysNotificationSetting
from app.repository.base import BaseRepository


class NotificationSettingRepository(BaseRepository[SysNotificationSetting]):
    model = SysNotificationSetting

    async def get_by_user_id(
        self,
        db: AsyncSession,
        user_id: int,
    ) -> Optional[SysNotificationSetting]:
        stmt = select(SysNotificationSetting).where(
            SysNotificationSetting.user_id == user_id
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def create(
        self,
        db: AsyncSession,
        entity: SysNotificationSetting,
    ) -> SysNotificationSetting:
        db.add(entity)
        await db.flush()
        await db.refresh(entity)
        return entity


notification_setting_repository = NotificationSettingRepository()
