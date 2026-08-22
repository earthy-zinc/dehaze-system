from fastapi import APIRouter, Body, Depends, File, Query, UploadFile
from fastapi.responses import Response
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
from app.service.algorithm_service import AlgorithmService

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
    algorithms = await AlgorithmService.get_algorithm_list(db, keywords)
    return success(algorithms)


@router.get("/options", response_model=Result[list[AlgorithmOptionVO]], summary="获取算法下拉选项")
async def get_algorithm_options(
    db: AsyncSession = Depends(get_db),
):
    options = await AlgorithmService.get_algorithm_options(db)
    return success(options)


@router.get("/list", response_model=Result[list[AlgorithmVO]], summary="获取所有算法扁平列表")
async def list_all_algorithms(
    db: AsyncSession = Depends(get_db),
):
    algorithms = await AlgorithmService.list_all_algorithms(db)
    return success(algorithms)


@router.get("/{algorithm_id}", response_model=Result[AlgorithmVO], summary="获取算法详情")
async def get_algorithm(
    algorithm_id: int,
    db: AsyncSession = Depends(get_db),
):
    algorithm = await AlgorithmService.get_algorithm_by_id(db, algorithm_id)
    return success(algorithm)


@router.post("", response_model=Result[int], summary="新增算法")
@require_permission("sys:algorithm:add")
async def create_algorithm(
    body: AlgorithmForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    algorithm_id = await AlgorithmService.create_algorithm(db, body.model_dump(exclude_none=True))
    return success(algorithm_id)


@router.put("/{algorithm_id}", response_model=Result[None], summary="修改算法")
@require_permission("sys:algorithm:edit")
async def update_algorithm(
    algorithm_id: int,
    body: AlgorithmForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await AlgorithmService.update_algorithm(db, algorithm_id, body.model_dump(exclude_none=True))
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
    await AlgorithmService.update_status(db, algorithm_id, status)
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
    await AlgorithmService.audit_algorithm(
        db=db,
        algorithm_id=algorithm_id,
        audit_by=user.id,
        passed=body.approved,
        remark=body.remark,
    )
    return success(msg="算法审核完成")


# ── 版本控制 ──────────────────────────────────────


@router.post("/{algorithm_id}/version", response_model=Result[int], summary="新增版本")
async def create_version(
    algorithm_id: int,
    body: AlgorithmVersionForm,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    algorithm_id = await AlgorithmService.create_version(
        db=db,
        algorithm_id=algorithm_id,
        version=body.version,
        change_log=body.changeLog,
        status=body.status,
        config_json=body.configJson,
        model_file_id=body.modelFileId,
        is_active=body.isActive or 0,
    )
    # 失效预测缓存
    from app.service.prediction_service import prediction_service

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
    versions = await AlgorithmService.list_versions(db, algorithm_id)
    return success(versions)


@router.post("/{algorithm_id}/rollback", response_model=Result[None], summary="版本回滚")
async def rollback_version(
    algorithm_id: int,
    versionId: int = Query(..., description="目标版本ID"),
    db: AsyncSession = Depends(get_db),
):
    await AlgorithmService.rollback_version(db, algorithm_id, versionId)
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
    await AlgorithmService.delete_algorithms(db, algorithm_ids)
    return success(msg="算法删除成功")


@router.delete("/{algorithm_id}", response_model=Result[None], summary="删除单个算法")
@require_permission("sys:algorithm:delete")
async def delete_algorithm_single(
    algorithm_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    """删除单个算法（含子算法）"""
    await AlgorithmService.delete_algorithm_single(db, algorithm_id)
    return success(msg="算法删除成功")


# ── 导入/导出 ──────────────────────────────────────


@router.get("/{algorithm_id}/_export", summary="导出单个算法（配置JSON）")
async def export_algorithm(
    algorithm_id: int,
    db: AsyncSession = Depends(get_db),
):
    """导出单个算法为 JSON 文件（对齐 Java exportAlgorithmJson）"""
    json_str = await AlgorithmService.export_algorithm(db, algorithm_id)
    filename = f"algorithm_{algorithm_id}.json"
    return Response(
        content=json_str.encode("utf-8"),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post(
    "/_import/validate",
    response_model=Result[str],
    summary="校验导入包",
)
async def validate_import_package(
    file: UploadFile = File(..., description="算法 JSON 文件"),
):
    """校验算法导入包格式，不写入数据库（对齐 Java validateImport）"""
    file_bytes = await file.read()
    message = await AlgorithmService.validate_import_package(file_bytes, file.filename or "")
    return success(message)


@router.post(
    "/_import",
    response_model=Result[None],
    summary="导入算法",
)
async def import_algorithm(
    file: UploadFile = File(..., description="算法 JSON 文件"),
    db: AsyncSession = Depends(get_db),
):
    """导入算法（对齐 Java importAlgorithm）"""
    file_bytes = await file.read()
    await AlgorithmService.import_algorithm(db, file_bytes, file.filename or "")
    return success(msg="算法导入成功")


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
    data = await AlgorithmService.get_monitor_data(db, algorithm_id)
    return success(data)


@router.get(
    "/{algorithm_id}/monitor/stats",
    summary="算法监控统计报表",
)
async def get_monitor_stats(
    algorithm_id: int,
    days: int = 7,
    db: AsyncSession = Depends(get_db),
):
    """获取算法监控统计报表（最近 days 天每天一条，含无数据天）"""
    data = await AlgorithmService.get_monitor_stats_report(db, algorithm_id, days)
    return success(data)
