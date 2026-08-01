from datetime import time
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
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

    async def upsert_by_user_id(
        self,
        db: AsyncSession,
        user_id: int,
    ) -> SysNotificationSetting:
        """upsert 通知设置：冲突时复活（重置 deleted=0）"""
        stmt = mysql_insert(SysNotificationSetting).values(
            user_id=user_id,
            push_enabled=1,
            dnd_enabled=0,
            dnd_start=time(22, 0),
            dnd_end=time(8, 0),
        )
        stmt = stmt.on_duplicate_key_update(
            deleted=0,
            update_time=func.now(),
        )
        await db.execute(stmt)
        result = await db.execute(
            select(SysNotificationSetting).where(
                SysNotificationSetting.user_id == user_id,
                SysNotificationSetting.deleted == 0,
            )
        )
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
