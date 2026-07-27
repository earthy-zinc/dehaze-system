from datetime import datetime, time
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_notification_setting import SysNotificationSetting
from app.repository.notification_setting_repository import notification_setting_repository

DEFAULT_PREFERENCES = {
    "typeChannels": {
        "announcement": {"push": True},
        "business": {"push": False},
        "member": {"push": True},
    },
    "moduleSwitches": {
        "prediction": True,
        "feedback": True,
        "announcement": True,
    },
}


def _format_time(t: Optional[time]) -> Optional[str]:
    if t is None:
        return None
    return t.strftime("%H:%M:%S")


def _parse_time(value: str) -> time:
    return datetime.strptime(value, "%H:%M:%S").time()


def _to_vo(setting: SysNotificationSetting) -> dict[str, Any]:
    return {
        "pushEnabled": bool(setting.push_enabled),
        "dndEnabled": bool(setting.dnd_enabled),
        "dndStart": _format_time(setting.dnd_start),
        "dndEnd": _format_time(setting.dnd_end),
        "preferences": setting.preferences or DEFAULT_PREFERENCES,
    }


def _deep_merge_preferences(
    old: Optional[dict], new: Optional[dict]
) -> dict[str, Any]:
    result = dict(old or DEFAULT_PREFERENCES)
    if not new:
        return result
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            merged = dict(result[key])
            merged.update(value)
            result[key] = merged
        else:
            result[key] = value
    return result


class NotificationSettingService:

    @staticmethod
    async def get_or_init(db: AsyncSession, user_id: int) -> dict[str, Any]:
        setting = await notification_setting_repository.get_by_user_id(db, user_id)
        if not setting:
            setting = SysNotificationSetting(
                user_id=user_id,
                push_enabled=1,
                dnd_enabled=0,
                dnd_start=time(22, 0),
                dnd_end=time(8, 0),
                preferences=DEFAULT_PREFERENCES,
            )
            await notification_setting_repository.create(db, setting)
        return _to_vo(setting)

    @staticmethod
    async def update(db: AsyncSession, user_id: int, data: dict[str, Any]) -> None:
        setting = await notification_setting_repository.get_by_user_id(db, user_id)
        if not setting:
            setting = SysNotificationSetting(
                user_id=user_id,
                push_enabled=1,
                dnd_enabled=0,
                dnd_start=time(22, 0),
                dnd_end=time(8, 0),
                preferences=DEFAULT_PREFERENCES,
            )
            await notification_setting_repository.create(db, setting)

        if "pushEnabled" in data:
            setting.push_enabled = 1 if data["pushEnabled"] else 0
        if "dndEnabled" in data:
            setting.dnd_enabled = 1 if data["dndEnabled"] else 0
        if "dndStart" in data and data["dndStart"]:
            setting.dnd_start = _parse_time(data["dndStart"])
        if "dndEnd" in data and data["dndEnd"]:
            setting.dnd_end = _parse_time(data["dndEnd"])
        if "preferences" in data and data["preferences"]:
            setting.preferences = _deep_merge_preferences(
                setting.preferences, data["preferences"]
            )
        await db.flush()
