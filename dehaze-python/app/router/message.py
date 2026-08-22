from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.message import MessageSendRequest
from app.service.message_service import MessageService

router = APIRouter(
    prefix="/api/v1/messages",
    tags=["消息通知"],
    dependencies=[Depends(get_current_user)],
)


@router.get("", summary="消息列表")
async def get_messages(
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=20, ge=1, le=100),
    type: str | None = Query(default=None),
    readStatus: int | None = Query(default=None, ge=0, le=1),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await MessageService.get_page(db, user.id, pageNum, pageSize, type, readStatus)
    return success(data)


@router.get("/unread-count", summary="未读消息数")
async def get_unread_count(
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    count = await MessageService.get_unread_count(db, user.id)
    return success({"count": count})


@router.get("/search", summary="搜索消息")
async def search_messages(
    keyword: str = Query(..., min_length=1),
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await MessageService.search(db, user.id, keyword, pageNum, pageSize)
    return success(data)


@router.patch("/_read-all", summary="全部标记已读")
async def mark_all_read(
    type: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    affected = await MessageService.mark_all_read(db, user.id, type)
    return success({"affectedCount": affected})


@router.post("/send", summary="内部消息发送")
async def send_message(
    body: MessageSendRequest,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    message_ids = await MessageService.send(db, body.model_dump())
    return success({"messageIds": message_ids})


@router.get("/{message_id}", summary="消息详情")
async def get_message_detail(
    message_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await MessageService.get_detail(db, user.id, message_id)
    return success(data)


@router.patch("/{message_id}/_read", summary="标记单条已读")
async def mark_read(
    message_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await MessageService.mark_read(db, user.id, message_id)
    return success()


@router.delete("/{ids}", summary="删除消息")
async def delete_messages(
    ids: str = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    try:
        id_list = [int(i) for i in ids.split(",")]
    except ValueError:
        id_list = []
    await MessageService.delete_by_ids(db, user.id, id_list)
    return success()
