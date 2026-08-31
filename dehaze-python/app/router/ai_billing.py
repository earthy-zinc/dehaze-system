"""AI 计费管理模块路由"""

from datetime import datetime

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import Result, success
from app.database import get_db
from app.decorators import require_permission
from app.decorators.permission import check_permission
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.ai_billing import (
    AdjustRequest,
    AnomalyRecordQuery,
    AnomalyRecordResult,
    BalanceResult,
    BillingRecordQuery,
    BillingRecordResult,
    BillingStatQuery,
    BillingStatResult,
    BillingSummaryResult,
    BillResult,
    CreditLogQuery,
    CreditLogResult,
    RefundAuditRequest,
    RefundCreateRequest,
    RefundQuery,
    RefundResult,
)
from app.models.schema.ai_billing_cost import (
    CostStatResult,
    ModelCostCreateRequest,
    ModelCostQuery,
    ModelCostResult,
    ModelCostUpdateRequest,
    ReconcileImportRequest,
)
from app.models.schema.common import PageResult
from app.service.billing.balance_service import balance_service
from app.service.billing.bill_service import bill_service
from app.service.billing.billing_anomaly_service import billing_anomaly_service
from app.service.billing.billing_record_service import billing_record_service
from app.service.billing.billing_stat_service import billing_stat_service
from app.service.billing.cost_service import cost_service
from app.service.billing.cost_stat_service import cost_stat_service
from app.service.billing.quota_service import quota_service
from app.service.billing.recharge_service import recharge_service
from app.service.billing.refund_service import refund_service

router = APIRouter(
    prefix="/api/v1/ai-billing",
    tags=["AI计费管理"],
    dependencies=[Depends(get_current_user)],
)


async def _build_balance(db: AsyncSession, user_id: int) -> BalanceResult:
    """组装余额账户视图"""
    daily_used, monthly_used = await quota_service.get_used(user_id)
    daily_limit, monthly_limit = await quota_service.get_limits(db, user_id)
    return BalanceResult(
        user_id=user_id,
        credits_balance=await balance_service.get_balance(db, user_id),
        arrears_status=await balance_service.is_arrears(user_id),
        daily_used=daily_used,
        daily_limit=daily_limit,
        monthly_used=monthly_used,
        monthly_limit=monthly_limit,
    )


def _resolve_query_user(user: UserContext, user_id_param: int | None) -> int:
    """解析查询目标用户：管理员可指定 userId 查询他人数据（需 ai:billing:stat），普通用户仅可查本人"""
    if user_id_param is None or user_id_param == user.id:
        return user.id
    if not check_permission(user, "ai:billing:stat"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ResultCode.ACCESS_UNAUTHORIZED.msg,
        )
    return user_id_param


# ==================== 用户端接口 ====================


@router.get("/balance", response_model=Result[BalanceResult], summary="用户余额查询")
async def get_balance(
    userId: int | None = Query(default=None, ge=1),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await _build_balance(db, _resolve_query_user(user, userId)))


@router.get("/summary", response_model=Result[BillingSummaryResult], summary="消耗汇总查询")
async def get_summary(
    dimension: str = Query(default="day"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await billing_stat_service.summary(db, user.id, dimension))


@router.get("/records", response_model=Result[PageResult[BillingRecordResult]], summary="计费明细查询")
async def list_records(
    userId: int | None = Query(default=None, ge=1),
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=20, ge=1, le=100),
    conversationId: int | None = Query(default=None),
    billType: str | None = Query(default=None),
    modelId: str | None = Query(default=None),
    dateStart: str | None = Query(default=None),
    dateEnd: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    query = BillingRecordQuery(
        page=pageNum,
        size=pageSize,
        conversation_id=conversationId,
        bill_type=billType,
        model_id=modelId,
        date_start=_parse_datetime(dateStart),
        date_end=_parse_datetime(dateEnd),
    )
    return success(
        await billing_record_service.list_by_user(db, _resolve_query_user(user, userId), query)
    )


@router.get("/credit-logs", response_model=Result[PageResult[CreditLogResult]], summary="余额流水查询")
async def list_credit_logs(
    userId: int | None = Query(default=None, ge=1),
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=20, ge=1, le=100),
    source: str | None = Query(default=None),
    dateStart: str | None = Query(default=None),
    dateEnd: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    query = CreditLogQuery(
        page=pageNum,
        size=pageSize,
        source=source,
        date_start=_parse_datetime(dateStart),
        date_end=_parse_datetime(dateEnd),
    )
    return success(
        await billing_record_service.list_credit_logs(db, _resolve_query_user(user, userId), query)
    )


@router.get("/bills/{month}", response_model=Result[BillResult], summary="月结账单查询")
async def get_bill(
    month: str = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await bill_service.get_bill(db, user.id, month))


@router.get("/bills/{month}/download", response_model=Result[BillResult], summary="账单下载")
async def download_bill(
    month: str = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    # 简化实现：返回账单 JSON，前端可另存为文件（保持 JSON 信封，与账单查询一致）
    return success(await bill_service.get_bill(db, user.id, month))


@router.post("/refunds", response_model=Result[RefundResult], summary="退款申请")
async def apply_refund(
    body: RefundCreateRequest = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(
        await refund_service.apply_refund(db, user.id, body.billing_id, body.amount, body.reason)
    )


# ==================== 管理员接口 ====================


@router.get("/refunds", response_model=Result[PageResult[RefundResult]], summary="退款申请列表")
@require_permission("ai:billing:refund")
async def list_refunds(
    userId: int | None = Query(default=None, ge=1),
    status: int | None = Query(default=None, ge=1, le=3),
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=20, ge=1, le=100),
    dateStart: str | None = Query(default=None),
    dateEnd: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    query = RefundQuery(
        page=pageNum,
        size=pageSize,
        status=status,
        user_id=userId,
        date_start=_parse_datetime(dateStart),
        date_end=_parse_datetime(dateEnd),
    )
    return success(await refund_service.list_refunds(db, query))


@router.get("/stats", response_model=Result[list[BillingStatResult]], summary="管理员计费统计")
@require_permission("ai:billing:stat")
async def get_stats(
    userId: int | None = Query(default=None),
    modelId: str | None = Query(default=None),
    billType: str | None = Query(default=None),
    dateStart: str | None = Query(default=None),
    dateEnd: str | None = Query(default=None),
    groupBy: str = Query(default="model"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    query = BillingStatQuery(
        group_by=groupBy,
        model_id=modelId,
        bill_type=billType,
        date_start=_parse_datetime(dateStart),
        date_end=_parse_datetime(dateEnd),
    )
    # user_id 额外透传（聚合维度需要）
    return success(await billing_stat_service.stats(db, query, user_id=userId))


@router.post("/adjust", response_model=Result[BalanceResult], summary="管理员手动调整积分")
@require_permission("ai:billing:adjust")
async def adjust_credits(
    body: AdjustRequest = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    if body.amount == 0:
        raise BusinessException(ResultCode.PARAM_ERROR, "调整积分数不能为 0")
    await recharge_service.recharge(
        db,
        body.user_id,
        body.amount,
        source="admin_adjust",
        reason=body.reason,
        operator_id=user.id,
    )
    return success(await _build_balance(db, body.user_id))


@router.post("/refunds/{refund_id}/audit", response_model=Result[RefundResult], summary="退款审核")
@require_permission("ai:billing:refund")
async def audit_refund(
    refund_id: int = Path(...),
    body: RefundAuditRequest = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(
        await refund_service.audit_refund(
            db, refund_id, body.approved, body.audit_remark, user.id
        )
    )


@router.get("/anomalies", response_model=Result[PageResult[AnomalyRecordResult]], summary="异常计费记录查询")
@require_permission("ai:billing:stat")
async def list_anomalies(
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=20, ge=1, le=100),
    userId: int | None = Query(default=None),
    anomalyType: str | None = Query(default=None),
    status: int | None = Query(default=None),
    dateStart: str | None = Query(default=None),
    dateEnd: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    query = AnomalyRecordQuery(
        user_id=userId,
        anomaly_type=anomalyType,
        status=status,
        date_start=_parse_datetime(dateStart),
        date_end=_parse_datetime(dateEnd),
        page=pageNum,
        size=pageSize,
    )
    return success(await billing_anomaly_service.list_anomalies(db, query))


@router.get("/costs", response_model=Result[PageResult[ModelCostResult]], summary="成本单价列表")
@require_permission("ai:billing:cost")
async def list_costs(
    keyword: str | None = Query(default=None),
    modelId: str | None = Query(default=None),
    providerId: int | None = Query(default=None),
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    query = ModelCostQuery(
        keyword=keyword,
        model_id=modelId,
        provider_id=providerId,
        page=pageNum,
        size=pageSize,
    )
    return success(await cost_service.list_costs(db, query))


@router.post("/costs", response_model=Result[ModelCostResult], summary="新增成本单价")
@require_permission("ai:billing:cost")
async def create_cost(
    body: ModelCostCreateRequest = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await cost_service.create_cost(db, body))


@router.put("/costs/{cost_id}", response_model=Result[ModelCostResult], summary="更新成本单价")
@require_permission("ai:billing:cost")
async def update_cost(
    cost_id: int = Path(...),
    body: ModelCostUpdateRequest = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = body.model_dump(exclude_unset=True, exclude_none=True)
    return success(await cost_service.update_cost(db, cost_id, data))


@router.delete("/costs/{cost_id}", response_model=Result[None], summary="删除成本单价")
@require_permission("ai:billing:cost")
async def delete_cost(
    cost_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await cost_service.delete_cost(db, cost_id)
    return success(None)


@router.get("/cost-stats", response_model=Result[list[CostStatResult]], summary="成本-利润统计")
@require_permission("ai:billing:cost")
async def get_cost_stats(
    startTime: str | None = Query(default=None),
    endTime: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(
        await cost_stat_service.cost_stats(
            db, _parse_datetime(startTime), _parse_datetime(endTime)
        )
    )


@router.post("/reconcile/import", response_model=Result[dict], summary="供应商账单导入")
@require_permission("ai:billing:cost")
async def import_reconcile(
    body: ReconcileImportRequest = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    imported = cost_service.import_reconcile(body.content)
    return success({"imported": imported})


def _parse_datetime(value: str | None) -> datetime | None:
    """解析前端传入的日期时间字符串，非法格式返回 None"""
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None
