"""热词管理路由（F-VS-004）"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import success
from app.database import get_db
from app.decorators.permission import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.voice import HotwordForm
from app.service.voice.hotword_service import HotwordService

router = APIRouter(prefix="/api/v1/voice/hotwords", tags=["热词管理"])


def _check_admin(user: UserContext) -> None:
    """管理员身份校验：非管理员抛出 A0301"""
    if not user.is_admin:
        raise BusinessException(ResultCode.ACCESS_UNAUTHORIZED, "仅管理员可操作")


@router.get("", summary="查询用户热词列表")
async def list_user_hotwords(
    db: AsyncSession = Depends(get_db),
    ctx: UserContext = Depends(get_current_user),
):
    return success(await HotwordService.list_user_hotwords(db, ctx.id))


@router.post("", summary="新增用户热词")
@require_permission("voice:hotword:edit")
async def add_user_hotword(
    form: HotwordForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await HotwordService.add_user_hotword(db, user.id, form))


@router.delete("/{hotword_id}", summary="删除用户热词")
@require_permission("voice:hotword:edit")
async def delete_user_hotword(
    hotword_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await HotwordService.delete_user_hotword(db, hotword_id, user.id)
    return success()


@router.get("/global", summary="查询全局热词列表")
async def list_global_hotwords(
    db: AsyncSession = Depends(get_db),
    ctx: UserContext = Depends(get_current_user),
):
    return success(await HotwordService.list_global_hotwords(db))


@router.post("/global", summary="新增全局热词（仅管理员）")
@require_permission("voice:hotword:edit")
async def add_global_hotword(
    form: HotwordForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    _check_admin(user)
    return success(await HotwordService.add_global_hotword(db, form))


@router.delete("/global/{hotword_id}", summary="删除全局热词（仅管理员）")
@require_permission("voice:hotword:edit")
async def delete_global_hotword(
    hotword_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    _check_admin(user)
    await HotwordService.delete_global_hotword(db, hotword_id)
    return success()
