from fastapi import APIRouter, Body, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db
from app.decorators.permission import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.algorithm import (
    AlgorithmAuditForm,
    AlgorithmForm,
    AlgorithmMonitorVO,
    AlgorithmOptionVO,
    AlgorithmVersionForm,
    AlgorithmVersionVO,
    AlgorithmVO,
)
from app.service.algorithm_service import algorithm_service

router = APIRouter(
    prefix="/api/v1/algorithms",
    tags=["算法管理"],
    dependencies=[Depends(get_current_user)],
)


@router.get("", response_model=Result[list[AlgorithmVO]], summary="获取算法树形表格")
async def list_algorithms(
    keywords: str | None = Query(default=None, description="关键词"),
    db: AsyncSession = Depends(get_db),
):
    algorithms = await algorithm_service.get_algorithm_list(db, keywords)
    return success(algorithms)


@router.get("/options", response_model=Result[list[AlgorithmOptionVO]], summary="获取算法下拉选项")
async def get_algorithm_options(
    db: AsyncSession = Depends(get_db),
):
    options = await algorithm_service.get_algorithm_options(db)
    return success(options)


@router.get("/list", response_model=Result[list[AlgorithmVO]], summary="获取所有算法扁平列表")
async def list_all_algorithms(
    db: AsyncSession = Depends(get_db),
):
    algorithms = await algorithm_service.list_all_algorithms(db)
    return success(algorithms)


@router.get("/{algorithm_id}", response_model=Result[AlgorithmVO], summary="获取算法详情")
async def get_algorithm(
    algorithm_id: int,
    db: AsyncSession = Depends(get_db),
):
    algorithm = await algorithm_service.get_algorithm_by_id(db, algorithm_id)
    return success(algorithm)


@router.post("", response_model=Result[int], summary="新增算法")
@require_permission("sys:algorithm:add")
async def create_algorithm(
    body: AlgorithmForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    algorithm_id = await algorithm_service.create_algorithm(db, body.model_dump(exclude_none=True))
    return success(algorithm_id)


@router.put("/{algorithm_id}", response_model=Result[None], summary="修改算法")
@require_permission("sys:algorithm:edit")
async def update_algorithm(
    algorithm_id: int,
    body: AlgorithmForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await algorithm_service.update_algorithm(db, algorithm_id, body.model_dump(exclude_none=True))
    return success(msg="算法更新成功")


# ── 状态机 ──────────────────────────────────────


@router.put("/{algorithm_id}/status", response_model=Result[None], summary="修改算法状态")
@require_permission("sys:algorithm:edit")
async def update_algorithm_status(
    algorithm_id: int,
    status: int = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await algorithm_service.update_status(db, algorithm_id, status)
    return success(msg="算法状态更新成功")


# ── 审核 ──────────────────────────────────────


@router.put("/{algorithm_id}/audit", response_model=Result[None], summary="审核算法")
@require_permission("sys:algorithm:audit")
async def audit_algorithm(
    algorithm_id: int,
    body: AlgorithmAuditForm,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await algorithm_service.audit_algorithm(
        db=db,
        algorithm_id=algorithm_id,
        audit_by=user.id,
        passed=body.approved,
        remark=body.remark,
    )
    return success(msg="算法审核完成")


# ── 版本控制 ──────────────────────────────────────


@router.post("/{algorithm_id}/version", response_model=Result[int], summary="新增版本")
@require_permission("sys:algorithm:version")
async def create_version(
    algorithm_id: int,
    body: AlgorithmVersionForm,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    algorithm_id = await algorithm_service.create_version(
        db=db,
        algorithm_id=algorithm_id,
        version=body.version,
        change_log=body.changeLog,
        config_json=body.configJson,
        model_file_id=body.modelFileId,
    )
    # 失效预测缓存
    from app.service.prediction.prediction_service import prediction_service

    await prediction_service.invalidate_cache(algorithm_id)
    return success(algorithm_id)


@router.get(
    "/{algorithm_id}/versions",
    response_model=Result[list[AlgorithmVersionVO]],
    summary="版本历史",
)
async def list_versions(
    algorithm_id: int,
    db: AsyncSession = Depends(get_db),
):
    """查询算法版本历史"""
    versions = await algorithm_service.list_versions(db, algorithm_id)
    return success(versions)


@router.post("/{algorithm_id}/rollback", response_model=Result[None], summary="版本回滚")
@require_permission("sys:algorithm:version")
async def rollback_version(
    algorithm_id: int,
    versionId: int = Query(..., description="目标版本ID"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await algorithm_service.rollback_version(db, algorithm_id, versionId)
    return success(msg="版本回滚成功")


# ── 删除 ──────────────────────────────────────


@router.delete("", response_model=Result[None], summary="批量删除算法")
@require_permission("sys:algorithm:delete")
async def delete_algorithms(
    ids: str = Query(..., description="算法ID，多个以逗号分隔"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    algorithm_ids = [int(i) for i in ids.split(",")]
    await algorithm_service.delete_algorithms(db, algorithm_ids)
    return success(msg="算法删除成功")


# ── 监控 ──────────────────────────────────────


@router.get(
    "/{algorithm_id}/monitor",
    response_model=Result[AlgorithmMonitorVO],
    summary="算法监控数据",
)
async def get_monitor_data(
    algorithm_id: int,
    db: AsyncSession = Depends(get_db),
):
    """获取算法实时监控数据（调用次数、平均耗时、成功率等）"""
    data = await algorithm_service.get_monitor_data(db, algorithm_id)
    return success(data)


@router.get(
    "/{algorithm_id}/monitor/stats",
    summary="算法监控统计报表",
)
async def get_monitor_stats(
    algorithm_id: int,
    days: int = Query(default=7, ge=1, description="统计天数（正整数）"),
    db: AsyncSession = Depends(get_db),
):
    """获取算法监控统计报表（最近 days 天每天一条，含无数据天）"""
    data = await algorithm_service.get_monitor_stats_report(db, algorithm_id, days)
    return success(data)
