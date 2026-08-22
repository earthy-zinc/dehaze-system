from fastapi import APIRouter, Body, Depends, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import success
from app.database import get_db
from app.decorators import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.member import (
    BenefitForm,
    MemberGrowthAdjustForm,
    MemberLevelAdjustForm,
    MemberStatusForm,
)
from app.service.member_service import MemberService

router = APIRouter(
    prefix="/api/v1/members",
    tags=["会员管理"],
    dependencies=[Depends(get_current_user)],
)


@router.get("/profile", summary="当前用户会员信息")
async def get_profile(
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await MemberService.get_profile(db, user.id)
    return success(data)


@router.get("/growth-logs", summary="成长值变动明细")
async def get_growth_logs(
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    changeType: str | None = Query(default=None),
    startTime: str | None = Query(default=None),
    endTime: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await MemberService.list_growth_logs(
        db,
        user.id,
        {
            "pageNum": pageNum,
            "pageSize": pageSize,
            "changeType": changeType,
            "startTime": startTime,
            "endTime": endTime,
        },
    )
    return success(data)


@router.post("/sign-in", summary="每日签到")
async def sign_in(
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await MemberService.sign_in(db, user.id)
    return success(data)


@router.get("/sign-in/calendar", summary="签到日历")
async def get_sign_in_calendar(
    year: int = Query(...),
    month: int = Query(..., ge=1, le=12),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await MemberService.get_sign_in_calendar(db, user.id, year, month)
    return success(data)


@router.get("/page", summary="会员分页列表")
@require_permission("member:list")
async def get_member_page(
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    keywords: str | None = Query(default=None),
    levelCode: str | None = Query(default=None),
    status: int | None = Query(default=None, ge=0, le=1),
    expireTimeStart: str | None = Query(default=None),
    expireTimeEnd: str | None = Query(default=None),
    growthMin: int | None = Query(default=None),
    growthMax: int | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await MemberService.list_paged_members(
        db,
        {
            "pageNum": pageNum,
            "pageSize": pageSize,
            "keywords": keywords,
            "levelCode": levelCode,
            "status": status,
            "expireTimeStart": expireTimeStart,
            "expireTimeEnd": expireTimeEnd,
            "growthMin": growthMin,
            "growthMax": growthMax,
        },
    )
    return success(data)


@router.get("/benefits", summary="权益配置列表")
async def list_benefits(
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await MemberService.list_benefits(db)
    return success(data)


@router.put("/benefits/{level_code}", summary="修改权益配置")
@require_permission("member:benefit:edit")
async def update_benefit(
    level_code: str = Path(...),
    body: BenefitForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await MemberService.update_benefit(db, level_code, body.model_dump(exclude_none=True))
    return success()


@router.get("/{user_id}", summary="会员详情")
async def get_member_detail(
    user_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await MemberService.get_member_detail(db, user_id)
    return success(data)


@router.put("/{user_id}/level", summary="等级调整")
@require_permission("member:level:edit")
async def adjust_level(
    user_id: int = Path(...),
    body: MemberLevelAdjustForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await MemberService.adjust_level(db, user_id, body.model_dump(), user.id)
    return success()


@router.put("/{user_id}/growth", summary="成长值调整")
@require_permission("member:growth:edit")
async def adjust_growth(
    user_id: int = Path(...),
    body: MemberGrowthAdjustForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await MemberService.adjust_growth(db, user_id, body.model_dump(), user.id)
    return success()


@router.put("/{user_id}/status", summary="冻结/解冻")
@require_permission("member:status:edit")
async def update_status(
    user_id: int = Path(...),
    body: MemberStatusForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await MemberService.update_status(db, user_id, body.model_dump())
    return success()
