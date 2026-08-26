from fastapi import APIRouter, Body, Depends, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import success
from app.database import get_db
from app.decorators import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.order import (
    AutoRenewConfigForm,
    BalanceRefundForm,
    OrderCreateForm,
    PayRequest,
    RefundApplyForm,
    RefundAuditForm,
)
from app.service.order.auto_renew_service import auto_renew_service
from app.service.order.order_service import order_service
from app.service.order.payment_service import payment_service
from app.service.order.refund_service import refund_service

router = APIRouter(
    prefix="/api/v1/orders",
    tags=["订单管理"],
    dependencies=[Depends(get_current_user)],
)


@router.post("", summary="创建订单")
async def create_order(
    body: OrderCreateForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await order_service.create(db, body.model_dump(exclude_none=True), user.id)
    return success(data)


@router.get("/balance", summary="查询余额账户")
async def get_balance(
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    from app.service.order.balance_account_service import balance_account_service

    data = await balance_account_service.get_balance(db, user.id)
    return success(data)


@router.post("/balance-refund", summary="提交余额退款申请")
async def apply_balance_refund(
    body: BalanceRefundForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await refund_service.apply_balance_refund(
        db, user.id, body.model_dump(exclude_none=True)
    )
    return success(data)


@router.get("/my", summary="我的订单列表")
async def list_my_orders(
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    status: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await order_service.list_my(
        db,
        user.id,
        {"pageNum": pageNum, "pageSize": pageSize, "status": status},
    )
    return success(data)


@router.get("/page", summary="订单分页列表")
@require_permission("order:list")
async def get_order_page(
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    orderNo: str | None = Query(default=None),
    keywords: str | None = Query(default=None),
    status: str | None = Query(default=None),
    packageType: str | None = Query(default=None),
    payMethod: str | None = Query(default=None),
    amountMin: int | None = Query(default=None, ge=0),
    amountMax: int | None = Query(default=None, ge=0),
    paidTimeStart: str | None = Query(default=None),
    paidTimeEnd: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await order_service.list_paged(
        db,
        {
            "pageNum": pageNum,
            "pageSize": pageSize,
            "orderNo": orderNo,
            "keywords": keywords,
            "status": status,
            "packageType": packageType,
            "payMethod": payMethod,
            "amountMin": amountMin,
            "amountMax": amountMax,
            "paidTimeStart": paidTimeStart,
            "paidTimeEnd": paidTimeEnd,
        },
        current_user=user,
    )
    return success(data)


@router.get("/refunds/page", summary="退款审核列表")
@require_permission("order:refund:list")
async def list_refunds(
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    orderNo: str | None = Query(default=None),
    keywords: str | None = Query(default=None),
    status: str | None = Query(default=None),
    reasonType: str | None = Query(default=None),
    applyTimeStart: str | None = Query(default=None),
    applyTimeEnd: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await refund_service.list_refunds(
        db,
        {
            "pageNum": pageNum,
            "pageSize": pageSize,
            "orderNo": orderNo,
            "keywords": keywords,
            "status": status,
            "reasonType": reasonType,
            "applyTimeStart": applyTimeStart,
            "applyTimeEnd": applyTimeEnd,
        },
    )
    return success(data)


@router.put("/refunds/{refund_id}/approve", summary="退款审核通过")
@require_permission("order:refund:approve")
async def approve_refund(
    refund_id: int = Path(...),
    body: RefundAuditForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await refund_service.approve_refund(db, refund_id, body.model_dump(), user.id)
    return success()


@router.put("/refunds/{refund_id}/reject", summary="退款审核驳回")
@require_permission("order:refund:approve")
async def reject_refund(
    refund_id: int = Path(...),
    body: RefundAuditForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await refund_service.reject_refund(db, refund_id, body.model_dump(), user.id)
    return success()


@router.get("/stats", summary="订单统计")
@require_permission("order:stats")
async def get_order_stats(
    startTime: str | None = Query(default=None),
    endTime: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await order_service.get_stats(db, startTime, endTime)
    return success(data)


@router.put("/auto-renew/config", summary="修改自动续费设置")
async def update_auto_renew_config(
    body: AutoRenewConfigForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await auto_renew_service.update_config(db, body.model_dump(), user.id)
    return success()


@router.get("/auto-renew/config", summary="查询自动续费配置")
async def get_auto_renew_config(
    packageId: int = Query(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await auto_renew_service.get_config(db, packageId, user.id)
    return success(data)


@router.get("/{order_no}", summary="订单详情")
async def get_order_detail(
    order_no: str = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await order_service.get_detail(db, order_no, user.id if not user.is_admin else None)
    return success(data)


@router.put("/{order_no}/cancel", summary="取消订单")
async def cancel_order(
    order_no: str = Path(...),
    reason: str = Query(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await order_service.cancel(db, order_no, reason, user.id)
    return success()


@router.post("/{order_no}/pay", summary="发起支付")
async def pay_order(
    order_no: str = Path(...),
    body: PayRequest = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await payment_service.pay(db, order_no, body.model_dump(), user.id)
    return success(data)


@router.post("/{order_no}/refund", summary="申请退款")
async def apply_refund(
    order_no: str = Path(...),
    body: RefundApplyForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await refund_service.apply_refund(db, order_no, body.model_dump(), user.id)
    return success()
