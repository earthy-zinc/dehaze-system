"""AI 可观测性查询路由（F-M08-013 后端实现 §2.6）

summary/traces/export/costs/trends 为管理端审计接口（ai:conversation:audit）；
过程链详情管理员全量可见，普通用户仅可查自己会话的过程链（A0401 不暴露存在性）。
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.result import Result, success
from app.database import get_db
from app.decorators.permission import check_permission
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.ai_observability import (
    CostsQuery,
    CostsResult,
    SummaryResult,
    TraceDetailResult,
    TraceItem,
    TracePageQuery,
    TrendsQuery,
    TrendItem,
)
from app.models.schema.common import PageResult
from app.service.ai_observability_service import ai_observability_service

router = APIRouter(prefix="/api/v1/ai", tags=["AI可观测性"])

_AUDIT_PERMISSION = "ai:conversation:audit"


def _require_audit(user: UserContext) -> None:
    """管理端审计权限校验：ROOT 放行，否则需 ai:conversation:audit（越权 A0301）"""
    if not check_permission(user, _AUDIT_PERMISSION):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ResultCode.FORBIDDEN_OPERATION.msg,
        )


@router.get(
    "/observability/summary",
    response_model=Result[SummaryResult],
    summary="异常总览统计",
)
async def get_summary(
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    _require_audit(user)
    return success(await ai_observability_service.summary(db))


@router.get(
    "/observability/traces",
    response_model=Result[PageResult[TraceItem]],
    summary="过程链检索",
)
async def list_traces(
    query: TracePageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    _require_audit(user)
    return success(await ai_observability_service.list_traces(db, query))


# 导出路由必须先于 /{trace_id} 注册，否则 export 会被当作路径参数解析
@router.get("/observability/traces/export", summary="过程链导出(CSV)")
async def export_traces(
    query: TracePageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    _require_audit(user)
    return await ai_observability_service.export_traces(db, query)


@router.get(
    "/observability/traces/{trace_id}",
    response_model=Result[TraceDetailResult],
    summary="过程链详情",
)
async def get_trace(
    trace_id: str,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    admin = check_permission(user, _AUDIT_PERMISSION)
    return success(
        await ai_observability_service.get_trace(db, trace_id, user.id, admin=admin)
    )


@router.get(
    "/observability/costs",
    response_model=Result[CostsResult],
    summary="资源消耗聚合",
)
async def get_costs(
    query: CostsQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    _require_audit(user)
    return success(await ai_observability_service.costs(db, query))


@router.get(
    "/observability/trends",
    response_model=Result[list[TrendItem]],
    summary="性能趋势",
)
async def get_trends(
    query: TrendsQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    _require_audit(user)
    return success(await ai_observability_service.trends(db, query))
