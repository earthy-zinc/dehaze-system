from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_announcement import SysAnnouncement
from app.models.entity.sys_dept import SysDept
from app.models.entity.sys_message import SysMessage
from app.models.entity.sys_user import SysUser
from app.repository.announcement_repository import announcement_repository
from app.repository.message_repository import message_repository

ANNOUNCEMENT_TYPE_LABELS = {
    "maintenance": "系统维护",
    "feature": "功能更新",
    "activity": "活动通知",
    "operation": "运营公告",
}

TARGET_SCOPE_LABELS = {
    "all": "全体用户",
    "level": "按会员等级",
    "tag": "按用户标签",
    "specified": "指定用户",
}

STATUS_LABELS = {
    1: "草稿",
    2: "待发送",
    3: "已发送",
    4: "已取消",
}

IMPORTANCE_LABELS = {1: "普通", 2: "重要"}


def _format_dt(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _parse_dt(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def _to_page_vo(a: SysAnnouncement) -> dict[str, Any]:
    return {
        "id": a.id,
        "title": a.title,
        "type": a.type,
        "typeLabel": ANNOUNCEMENT_TYPE_LABELS.get(a.type, ""),
        "importance": a.importance,
        "targetScope": a.target_scope,
        "targetScopeLabel": TARGET_SCOPE_LABELS.get(a.target_scope, ""),
        "status": a.status,
        "statusLabel": STATUS_LABELS.get(a.status, ""),
        "sendTime": _format_dt(a.send_time),
        "expireTime": _format_dt(a.expire_time),
        "sentCount": a.sent_count,
        "createTime": _format_dt(a.create_time),
        "createBy": a.create_by,
    }


def _to_detail_vo(a: SysAnnouncement) -> dict[str, Any]:
    vo = _to_page_vo(a)
    vo["content"] = a.content
    vo["importanceLabel"] = IMPORTANCE_LABELS.get(a.importance, "")
    vo["targetParams"] = a.target_params
    vo["updateTime"] = _format_dt(a.update_time)
    return vo


class AnnouncementService:

    @staticmethod
    async def create(db: AsyncSession, data: dict[str, Any], user_id: int) -> int:
        send_time = data.get("sendTime")
        status = 2 if send_time else 1

        announcement = SysAnnouncement(
            title=data["title"],
            content=data["content"],
            type=data["type"],
            importance=data["importance"],
            target_scope=data["targetScope"],
            target_params=data.get("targetParams"),
            status=status,
            send_time=_parse_dt(send_time) if send_time else None,
            expire_time=_parse_dt(data["expireTime"]) if data.get("expireTime") else None,
            sent_count=0,
            deleted=0,
        )
        await announcement_repository.create(db, announcement)
        return announcement.id

    @staticmethod
    async def get_page(
        db: AsyncSession,
        page: int,
        page_size: int,
        title: Optional[str] = None,
        type: Optional[str] = None,
        status: Optional[int] = None,
    ) -> dict[str, Any]:
        items, total = await announcement_repository.get_page(
            db, page, page_size, title, type, status
        )
        list_data = [_to_page_vo(a) for a in items]
        return {"list": list_data, "total": total, "pageNum": page, "pageSize": page_size}

    @staticmethod
    async def get_detail(db: AsyncSession, announcement_id: int) -> dict[str, Any]:
        announcement = await announcement_repository.get_by_id(db, announcement_id)
        if not announcement:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "公告不存在")
        return _to_detail_vo(announcement)

    @staticmethod
    async def update(db: AsyncSession, announcement_id: int, data: dict[str, Any]) -> None:
        announcement = await announcement_repository.get_by_id(db, announcement_id)
        if not announcement:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "公告不存在")
        if announcement.status not in (1, 2):
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "公告状态不允许编辑")

        if "title" in data:
            announcement.title = data["title"]
        if "content" in data:
            announcement.content = data["content"]
        if "type" in data:
            announcement.type = data["type"]
        if "importance" in data:
            announcement.importance = data["importance"]
        if "targetScope" in data:
            announcement.target_scope = data["targetScope"]
        if "targetParams" in data:
            announcement.target_params = data["targetParams"]
        if "sendTime" in data:
            announcement.send_time = _parse_dt(data["sendTime"]) if data["sendTime"] else None
        if "expireTime" in data:
            announcement.expire_time = _parse_dt(data["expireTime"]) if data["expireTime"] else None
        await db.flush()

    @staticmethod
    async def delete(db: AsyncSession, announcement_id: int) -> None:
        announcement = await announcement_repository.get_by_id(db, announcement_id)
        if not announcement:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "公告不存在")
        await announcement_repository.soft_delete(db, announcement_id)

    @staticmethod
    async def send(db: AsyncSession, announcement_id: int) -> int:
        announcement = await announcement_repository.get_by_id(db, announcement_id)
        if not announcement:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "公告不存在")
        if announcement.status not in (1, 2):
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "公告状态不允许发送")

        target_user_ids = await AnnouncementService._resolve_targets(
            db, announcement.target_scope, announcement.target_params
        )

        priority = 3 if announcement.importance == 2 else 2
        expires_at = datetime.now() + timedelta(days=30)

        messages = [
            SysMessage(
                type="announcement",
                title=announcement.title,
                content=announcement.content,
                sender_type=2,
                recipient_id=uid,
                biz_module="system",
                biz_id=str(announcement.id),
                priority=priority,
                jump_url=None,
                extra=None,
                read_status=0,
                deleted=0,
                expires_at=expires_at,
            )
            for uid in target_user_ids
        ]

        if messages:
            await message_repository.batch_create(db, messages)

        await announcement_repository.update_status(
            db,
            announcement_id,
            status=3,
            sent_count=len(target_user_ids),
            send_time=datetime.now(),
        )
        return len(target_user_ids)

    @staticmethod
    async def cancel(db: AsyncSession, announcement_id: int) -> None:
        announcement = await announcement_repository.get_by_id(db, announcement_id)
        if not announcement:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "公告不存在")
        if announcement.status != 2:
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "公告状态不允许取消")
        await announcement_repository.update_status(db, announcement_id, status=4)

    @staticmethod
    async def _resolve_targets(
        db: AsyncSession,
        target_scope: str,
        target_params: Optional[dict],
    ) -> list[int]:
        if target_scope == "all":
            stmt = select(SysUser.id).where(
                SysUser.deleted == 0, SysUser.status == 1
            )
            result = await db.execute(stmt)
            return [row[0] for row in result.fetchall()]

        if target_scope == "specified":
            if not target_params:
                return []
            user_ids = target_params.get("userIds") or []
            return [int(uid) for uid in user_ids]

        if target_scope == "level":
            if not target_params:
                return []
            level = target_params.get("level")
            if level is None:
                return []
            level_code = f"level_{level}"
            stmt = text(
                "SELECT user_id FROM sys_member WHERE level_code = :level_code AND deleted = 0 AND status = 1"
            )
            result = await db.execute(stmt, {"level_code": level_code})
            return [row[0] for row in result.fetchall()]

        if target_scope == "tag":
            if not target_params:
                return []
            tags = target_params.get("tags")
            if not tags:
                tag = target_params.get("tag")
                tags = [tag] if tag else []
            if not tags:
                return []
            stmt = (
                select(SysUser.id)
                .join(SysDept, SysUser.dept_id == SysDept.id)
                .where(
                    SysUser.deleted == 0,
                    SysUser.status == 1,
                    SysDept.deleted == 0,
                    SysDept.status == 1,
                    SysDept.name.in_(tags),
                )
            )
            result = await db.execute(stmt)
            return [row[0] for row in result.fetchall()]

        return []
