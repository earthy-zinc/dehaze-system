from datetime import datetime, time
from typing import Any

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


def _format_time(t: time | None) -> str | None:
    if t is None:
        return None
    return t.strftime("%H:%M:%S")


def _to_vo(setting: SysNotificationSetting) -> dict[str, Any]:
    return {
        "pushEnabled": bool(setting.push_enabled),
        "dndEnabled": bool(setting.dnd_enabled),
        "dndStart": _format_time(setting.dnd_start),
        "dndEnd": _format_time(setting.dnd_end),
        "preferences": setting.preferences or DEFAULT_PREFERENCES,
    }


def _deep_merge_preferences(old: dict | None, new: dict | None) -> dict[str, Any]:
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
    async def get_or_init(self, db: AsyncSession, user_id: int) -> dict[str, Any]:
        setting = await notification_setting_repository.get_by_user_id(db, user_id)
        if not setting:
            setting = await notification_setting_repository.upsert_by_user_id(db, user_id)
        return _to_vo(setting)

    async def update(self, db: AsyncSession, user_id: int, data: dict[str, Any]) -> None:
        setting = await notification_setting_repository.get_by_user_id(db, user_id)
        if not setting:
            setting = await notification_setting_repository.upsert_by_user_id(db, user_id)

        if "pushEnabled" in data:
            setting.push_enabled = 1 if data["pushEnabled"] else 0
        if "dndEnabled" in data:
            setting.dnd_enabled = 1 if data["dndEnabled"] else 0
        if "dndStart" in data and data["dndStart"]:
            setting.dnd_start = datetime.strptime(data["dndStart"], "%H:%M:%S").time()
        if "dndEnd" in data and data["dndEnd"]:
            setting.dnd_end = datetime.strptime(data["dndEnd"], "%H:%M:%S").time()
        if "preferences" in data and data["preferences"]:
            setting.preferences = _deep_merge_preferences(setting.preferences, data["preferences"])
        await db.flush()


# 单例
notification_setting_service = NotificationSettingService()
