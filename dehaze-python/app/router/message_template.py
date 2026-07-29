from typing import Optional

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.message import MessageTemplateForm
from app.service.message_template_service import MessageTemplateService

router = APIRouter(
    prefix="/api/v1/message-templates",
    tags=["消息模板"],
    dependencies=[Depends(get_current_user)],
)


@router.get("/page", summary="模板分页列表")
async def get_template_page(
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=20, ge=1, le=100),
    name: Optional[str] = Query(default=None),
    type: Optional[str] = Query(default=None),
    status: Optional[int] = Query(default=None, ge=0, le=1),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await MessageTemplateService.get_page(db, pageNum, pageSize, name, type, status)
    return success(data)


@router.get("/{template_id}", summary="模板详情")
async def get_template_detail(
    template_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await MessageTemplateService.get_detail(db, template_id)
    return success(data)


@router.put("/{template_id}", summary="编辑模板")
@require_permission("notify:template:edit")
async def update_template(
    template_id: int = Path(...),
    body: MessageTemplateForm = None,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await MessageTemplateService.update(db, template_id, body.model_dump(exclude_none=True) if body else {})
    return success()
