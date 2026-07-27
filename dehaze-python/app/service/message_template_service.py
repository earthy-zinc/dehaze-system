from datetime import datetime
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_message_template import SysMessageTemplate
from app.repository.message_template_repository import message_template_repository


def _format_dt(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


class MessageTemplateService:

    @staticmethod
    async def get_page(
        db: AsyncSession,
        page: int,
        page_size: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        status: Optional[int] = None,
    ) -> dict[str, Any]:
        items, total = await message_template_repository.get_page(
            db, page, page_size, name, type, status
        )
        list_data = [
            {
                "id": t.id,
                "code": t.code,
                "name": t.name,
                "type": t.type,
                "titleTemplate": t.title_template,
                "priority": t.priority,
                "status": t.status,
                "createTime": _format_dt(t.create_time),
            }
            for t in items
        ]
        return {"list": list_data, "total": total, "pageNum": page, "pageSize": page_size}

    @staticmethod
    async def get_detail(db: AsyncSession, template_id: int) -> dict[str, Any]:
        template = await message_template_repository.get_by_id(db, template_id)
        if not template:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "模板不存在")
        return {
            "id": template.id,
            "code": template.code,
            "name": template.name,
            "type": template.type,
            "titleTemplate": template.title_template,
            "contentTemplate": template.content_template,
            "priority": template.priority,
            "channels": template.channels,
            "variables": template.variables,
            "status": template.status,
            "createTime": _format_dt(template.create_time),
            "updateTime": _format_dt(template.update_time),
        }

    @staticmethod
    async def update(db: AsyncSession, template_id: int, data: dict[str, Any]) -> None:
        template = await message_template_repository.get_by_id(db, template_id)
        if not template:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "模板不存在")

        if "name" in data:
            template.name = data["name"]
        if "titleTemplate" in data:
            template.title_template = data["titleTemplate"]
        if "contentTemplate" in data:
            template.content_template = data["contentTemplate"]
        if "priority" in data:
            template.priority = data["priority"]
        if "channels" in data:
            template.channels = data["channels"]
        if "status" in data:
            template.status = data["status"]
        await db.flush()
