from fastapi import APIRouter, Body, Depends, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import success
from app.database import get_db
from app.decorators import require_permission
from app.decorators.permission import check_permission
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.member import (
    BenefitForm,
    MemberGrowthAdjustForm,
    MemberLevelAdjustForm,
    MemberStatusForm,
)
from app.service.member.benefit_service import member_benefit_service
from app.service.member.growth_service import member_growth_service
from app.service.member.member_service import member_service
from app.service.order.order_service import order_service

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
    data = await member_service.get_profile(db, user.id)
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
    data = await member_growth_service.list_growth_logs(
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
    data = await member_growth_service.sign_in(db, user.id)
    return success(data)


@router.get("/sign-in/calendar", summary="签到日历")
async def get_sign_in_calendar(
    year: int = Query(...),
    month: int = Query(..., ge=1, le=12),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await member_growth_service.get_sign_in_calendar(db, user.id, year, month)
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
    data = await member_service.list_paged_members(
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
    data = await member_benefit_service.list_benefits(db)
    return success(data)


@router.put("/benefits/{level_code}", summary="修改权益配置")
@require_permission("member:benefit:edit")
async def update_benefit(
    level_code: str = Path(...),
    body: BenefitForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await member_benefit_service.update_benefit(db, level_code, body.model_dump(exclude_none=True))
    return success()


@router.get("/benefit-summary", summary="当前用户权益概览")
async def get_benefit_summary(
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await member_service.get_benefit_summary(db, user.id)
    return success(data)


@router.get("/trial-status", summary="当前用户试用引导状态")
async def get_trial_status(
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await member_service.get_trial_status(db, user.id)
    return success(data)


@router.get("/{user_id}", summary="会员详情")
async def get_member_detail(
    user_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    # 默认仅本人可见；持 member:list 权限可查任意会员（管理端列表/详情弹窗入口）
    if user.id != user_id and not check_permission(user, "member:list"):
        raise BusinessException(ResultCode.ACCESS_UNAUTHORIZED, "无权查看他人会员详情")
    data = await member_service.get_member_detail(db, user_id)
    return success(data)


@router.get("/{user_id}/growth-logs", summary="会员成长值流水（管理端）")
@require_permission("member:list")
async def get_member_growth_logs(
    user_id: int = Path(...),
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    changeType: str | None = Query(default=None),
    startTime: str | None = Query(default=None),
    endTime: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await member_growth_service.list_growth_logs(
        db,
        user_id,
        {
            "pageNum": pageNum,
            "pageSize": pageSize,
            "changeType": changeType,
            "startTime": startTime,
            "endTime": endTime,
        },
    )
    return success(data)


@router.get("/{user_id}/consumption-records", summary="会员消费记录（管理端）")
@require_permission("member:list")
async def get_member_consumption_records(
    user_id: int = Path(...),
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    status: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await order_service.list_my(
        db, user_id, {"pageNum": pageNum, "pageSize": pageSize, "status": status}
    )
    return success(data)


@router.get("/{user_id}/benefit-usage", summary="会员权益使用明细（管理端）")
@require_permission("member:list")
async def get_member_benefit_usage(
    user_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await member_service.get_benefit_summary(db, user_id)
    return success(data)


@router.get("/{user_id}/operation-logs", summary="会员操作日志（管理端）")
@require_permission("member:list")
async def get_member_operation_logs(
    user_id: int = Path(...),
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    user: UserContext = Depends(get_current_user),
):
    data = await member_service.list_member_audit_logs(user_id, pageNum, pageSize)
    return success(data)


@router.put("/{user_id}/level", summary="等级调整")
@require_permission("member:level:edit")
async def adjust_level(
    user_id: int = Path(...),
    body: MemberLevelAdjustForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await member_service.adjust_level(db, user_id, body.model_dump(), user.id)
    return success()


@router.put("/{user_id}/growth", summary="成长值调整")
@require_permission("member:growth:edit")
async def adjust_growth(
    user_id: int = Path(...),
    body: MemberGrowthAdjustForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await member_service.adjust_growth(db, user_id, body.model_dump(), user.id)
    return success()


@router.put("/{user_id}/status", summary="冻结/解冻")
@require_permission("member:status:edit")
async def update_status(
    user_id: int = Path(...),
    body: MemberStatusForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await member_service.update_status(db, user_id, body.model_dump(), user.id)
    return success()
