import logging
import re
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.redis import get_redis_client
from app.models.entity.sys_message import SysMessage
from app.models.entity.sys_user import SysUser
from app.repository.message_repository import message_repository
from app.repository.message_template_repository import message_template_repository

TYPE_LABELS = {
    "inbox": "站内信",
    "announcement": "系统公告",
    "business": "业务通知",
    "member": "会员通知",
    "alert": "告警通知",
    "critical_alert": "严重告警",
}

EXPIRY_DAYS = {
    "alert": 7,
    "critical_alert": 90,
}

VAR_PATTERN = re.compile(r"\{(\w+)\}")

logger = logging.getLogger(__name__)

UNREAD_COUNT_CACHE_PREFIX = "msg:unread:"
UNREAD_COUNT_CACHE_TTL = 3600


async def invalidate_unread_count_cache(*user_ids: int) -> None:
    if not user_ids:
        return
    keys = [f"{UNREAD_COUNT_CACHE_PREFIX}{uid}" for uid in user_ids]
    try:
        redis = await get_redis_client()
        await redis.delete(*keys)
    except Exception as e:
        logger.warning("失效未读数缓存失败: user_ids=%s err=%s", user_ids, e)


async def _push_new_message_event(message: SysMessage) -> None:
    try:
        from app.service.websocket_service import manager as ws_manager
        await ws_manager.send_personal(message.recipient_id, {
            "event": "new_message",
            "data": {
                "id": message.id,
                "type": message.type,
                "title": message.title,
                "priority": message.priority,
                "createTime": _format_dt(message.create_time),
            },
        })
    except Exception as e:
        logger.debug("WebSocket 推送新消息事件失败（不影响主流程）: messageId=%s err=%s", message.id, e)


def _format_dt(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _calc_expires_at(msg_type: str) -> datetime:
    days = EXPIRY_DAYS.get(msg_type, 30)
    return datetime.now() + timedelta(days=days)


def _render_template(template: str, variables: dict[str, str]) -> str:
    def replacer(m):
        return variables.get(m.group(1), "")
    return VAR_PATTERN.sub(replacer, template)


def _extract_var_names(template_str: str) -> list[str]:
    return VAR_PATTERN.findall(template_str)


class MessageService:

    @staticmethod
    async def send(db: AsyncSession, data: dict[str, Any]) -> list[int]:
        biz_module = data.get("bizModule")
        biz_id = data.get("bizId")

        recipient_ids = data["recipientIds"]
        message_ids: list[int] = []

        if biz_module and biz_id:
            existing = await message_repository.find_by_biz_and_recipients(
                db, biz_module, biz_id, recipient_ids
            )
            existing_map = {m.recipient_id: m.id for m in existing}
            remaining = []
            for rid in recipient_ids:
                if rid in existing_map:
                    message_ids.append(existing_map[rid])
                else:
                    remaining.append(rid)
            recipient_ids = remaining

        if not recipient_ids:
            return message_ids

        template_code = data.get("templateCode")
        msg_type = data["type"]
        title = data.get("title")
        content = data.get("content")
        priority = data.get("priority", 2)

        if template_code:
            template = await message_template_repository.get_by_code(db, template_code)
            if not template:
                raise BusinessException(ResultCode.MESSAGE_TEMPLATE_NOT_FOUND, "模板不存在")
            if template.status == 0:
                raise BusinessException(ResultCode.TEMPLATE_DISABLED, "模板已禁用")

            variables = data.get("variables") or {}
            required_vars = set(_extract_var_names(template.title_template or ""))
            required_vars |= set(_extract_var_names(template.content_template or ""))

            missing = required_vars - set(variables.keys())
            if missing:
                raise BusinessException(ResultCode.TEMPLATE_VAR_MISSING, f"模板变量缺失: {','.join(missing)}")

            title = _render_template(template.title_template, variables)
            content = _render_template(template.content_template, variables)
            if priority == 2 and template.priority:
                priority = template.priority
        else:
            if not title:
                raise BusinessException(ResultCode.PARAM_ERROR, "消息标题不能为空")
            if not content:
                raise BusinessException(ResultCode.PARAM_ERROR, "消息正文不能为空")

        jump_url = data.get("jumpUrl")
        extra = data.get("extra")
        expires_at = _calc_expires_at(msg_type)

        messages = [
            SysMessage(
                type=msg_type,
                title=title,
                content=content,
                sender_type=1,
                recipient_id=rid,
                biz_module=biz_module,
                biz_id=biz_id,
                priority=priority,
                jump_url=jump_url,
                extra=extra,
                read_status=0,
                deleted=0,
                expires_at=expires_at,
            )
            for rid in recipient_ids
        ]
        await message_repository.batch_create(db, messages)
        await invalidate_unread_count_cache(*recipient_ids)
        for m in messages:
            await _push_new_message_event(m)
        message_ids.extend(m.id for m in messages)
        return message_ids

    @staticmethod
    async def get_page(
        db: AsyncSession,
        user_id: int,
        page: int,
        page_size: int,
        type: Optional[str] = None,
        read_status: Optional[int] = None,
    ) -> dict[str, Any]:
        items, total = await message_repository.get_page(
            db, user_id, page, page_size, type, read_status
        )
        list_data = [
            {
                "id": m.id,
                "type": m.type,
                "typeLabel": TYPE_LABELS.get(m.type, ""),
                "title": m.title,
                "summary": (m.content[:50] if m.content else ""),
                "priority": m.priority,
                "readStatus": m.read_status,
                "senderType": m.sender_type,
                "jumpUrl": m.jump_url,
                "createTime": _format_dt(m.create_time),
            }
            for m in items
        ]
        return {"list": list_data, "total": total, "pageNum": page, "pageSize": page_size}

    @staticmethod
    async def get_unread_count(db: AsyncSession, user_id: int) -> int:
        cache_key = f"{UNREAD_COUNT_CACHE_PREFIX}{user_id}"
        try:
            redis = await get_redis_client()
            cached = await redis.get(cache_key)
            if cached is not None:
                return int(cached)
        except Exception as e:
            logger.warning("读取未读数缓存失败: user_id=%s err=%s", user_id, e)
        count = await message_repository.count_unread(db, user_id)
        try:
            redis = await get_redis_client()
            ttl = 300 if count == 0 else UNREAD_COUNT_CACHE_TTL
            await redis.set(cache_key, str(count), ex=ttl)
        except Exception as e:
            logger.warning("写入未读数缓存失败: user_id=%s err=%s", user_id, e)
        return count

    @staticmethod
    async def get_detail(db: AsyncSession, user_id: int, message_id: int) -> dict[str, Any]:
        msg = await message_repository.get_by_id_and_recipient(db, message_id, user_id)
        if not msg:
            raise BusinessException(ResultCode.MESSAGE_NOT_FOUND, "消息不存在")

        return {
            "id": msg.id,
            "type": msg.type,
            "typeLabel": TYPE_LABELS.get(msg.type, ""),
            "title": msg.title,
            "content": msg.content,
            "priority": msg.priority,
            "senderType": msg.sender_type,
            "readStatus": msg.read_status,
            "readTime": _format_dt(msg.read_time),
            "jumpUrl": msg.jump_url,
            "extra": msg.extra,
            "createTime": _format_dt(msg.create_time),
        }

    @staticmethod
    async def mark_read(db: AsyncSession, user_id: int, message_id: int) -> None:
        msg = await message_repository.get_by_id_and_recipient(db, message_id, user_id)
        if not msg:
            raise BusinessException(ResultCode.MESSAGE_NOT_FOUND, "消息不存在")
        if msg.read_status == 0:
            await message_repository.mark_read(db, message_id, user_id)
            await invalidate_unread_count_cache(user_id)

    @staticmethod
    async def mark_all_read(
        db: AsyncSession,
        user_id: int,
        type: Optional[str] = None,
    ) -> int:
        affected = await message_repository.mark_all_read(db, user_id, type)
        await invalidate_unread_count_cache(user_id)
        return affected

    @staticmethod
    async def delete_by_ids(db: AsyncSession, user_id: int, ids: list[int]) -> None:
        await message_repository.soft_delete_by_ids_and_recipient(db, ids, user_id)
        await invalidate_unread_count_cache(user_id)

    @staticmethod
    async def search(
        db: AsyncSession,
        user_id: int,
        keyword: str,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        items, total = await message_repository.search(
            db, user_id, keyword, page, page_size
        )
        list_data = [
            {
                "id": m.id,
                "type": m.type,
                "typeLabel": TYPE_LABELS.get(m.type, ""),
                "title": m.title,
                "summary": (m.content[:50] if m.content else ""),
                "priority": m.priority,
                "readStatus": m.read_status,
                "senderType": m.sender_type,
                "jumpUrl": m.jump_url,
                "createTime": _format_dt(m.create_time),
            }
            for m in items
        ]
        return {"list": list_data, "total": total, "pageNum": page, "pageSize": page_size}

    @staticmethod
    async def refresh_unread_count_cache(db: AsyncSession) -> int:
        stmt = select(SysUser.id).where(
            SysUser.deleted == 0,
            SysUser.status == 1,
        )
        result = await db.execute(stmt)
        user_ids = [row[0] for row in result.fetchall()]

        if not user_ids:
            logger.debug("未读数缓存刷新: 无活跃用户")
            return 0

        redis = None
        try:
            redis = await get_redis_client()
        except Exception as e:
            logger.warning("Redis 不可用，未读数缓存刷新跳过: %s", e)
            return 0

        refreshed = 0
        for user_id in user_ids:
            count = await message_repository.count_unread(db, user_id)
            cache_key = f"{UNREAD_COUNT_CACHE_PREFIX}{user_id}"
            ttl = 300 if count == 0 else UNREAD_COUNT_CACHE_TTL
            try:
                await redis.set(cache_key, str(count), ex=ttl)
                refreshed += 1
            except Exception as e:
                logger.warning("写入未读数缓存失败: user_id=%s err=%s", user_id, e)

        logger.debug("未读数缓存刷新完成: 共刷新 %s 个用户", refreshed)
        return refreshed
