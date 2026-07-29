from typing import Optional

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import success
from app.database import get_db
from app.decorators import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.message import AnnouncementForm, AnnouncementUpdateForm
from app.service.announcement_service import AnnouncementService

router = APIRouter(
    prefix="/api/v1/announcements",
    tags=["系统公告"],
    dependencies=[Depends(get_current_user)],
)


@router.get("/page", summary="公告分页列表")
async def get_announcement_page(
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    title: Optional[str] = Query(default=None),
    type: Optional[str] = Query(default=None),
    status: Optional[int] = Query(default=None, ge=1, le=4),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await AnnouncementService.get_page(db, pageNum, pageSize, title, type, status)
    return success(data)


@router.post("", summary="创建公告")
@require_permission("notify:announcement:add")
async def create_announcement(
    body: AnnouncementForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    announcement_id = await AnnouncementService.create(db, body.model_dump(), user.id)
    return success({"id": announcement_id})


@router.get("/{announcement_id}", summary="公告详情")
async def get_announcement_detail(
    announcement_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await AnnouncementService.get_detail(db, announcement_id)
    return success(data)


@router.put("/{announcement_id}", summary="编辑公告")
@require_permission("notify:announcement:edit")
async def update_announcement(
    body: AnnouncementUpdateForm,
    announcement_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await AnnouncementService.update(db, announcement_id, body.model_dump(exclude_unset=True))
    return success()


@router.delete("/{announcement_id}", summary="删除公告")
@require_permission("notify:announcement:delete")
async def delete_announcement(
    announcement_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await AnnouncementService.delete(db, announcement_id)
    return success()


@router.post("/{announcement_id}/_send", summary="发送公告")
@require_permission("notify:announcement:send")
async def send_announcement(
    announcement_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    sent_count = await AnnouncementService.send(db, announcement_id)
    return success({"sentCount": sent_count})


@router.patch("/{announcement_id}/_cancel", summary="取消定时公告")
@require_permission("notify:announcement:cancel")
async def cancel_announcement(
    announcement_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await AnnouncementService.cancel(db, announcement_id)
    return success()
