from typing import Optional

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.result import Result, error, success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.algorithm import (
    AlgorithmAuditForm,
    AlgorithmDeleteResultVO,
    AlgorithmForm,
    AlgorithmIdVO,
    AlgorithmImportResultVO,
    AlgorithmMonitorVO,
    AlgorithmMonitorStatsVO,
    AlgorithmOptionVO,
    AlgorithmRollbackForm,
    AlgorithmStatusForm,
    AlgorithmVersionForm,
    AlgorithmVersionVO,
    AlgorithmVO,
)
from app.service.algorithm_service import AlgorithmService

router = APIRouter(
    prefix="/api/v1/algorithm",
    tags=["算法管理"],
    dependencies=[Depends(get_current_user)],
)


@router.get("", response_model=Result[list[AlgorithmVO]], summary="获取算法树形表格")
async def list_algorithms(
    keywords: Optional[str] = Query(default=None, description="关键词"),
    db: AsyncSession = Depends(get_db),
):
    algorithms = await AlgorithmService.get_algorithm_list(db, keywords)
    return success(algorithms)


@router.get(
    "/options", response_model=Result[list[AlgorithmOptionVO]], summary="获取算法下拉选项"
)
async def get_algorithm_options(
    db: AsyncSession = Depends(get_db),
):
    options = await AlgorithmService.get_algorithm_options(db)
    return success(options)


@router.get(
    "/{algorithm_id}", response_model=Result[AlgorithmVO], summary="获取算法详情"
)
async def get_algorithm(
    algorithm_id: int,
    db: AsyncSession = Depends(get_db),
):
    algorithm = await AlgorithmService.get_algorithm_by_id(db, algorithm_id)
    if algorithm:
        return success(algorithm)
    return error("算法不存在", ResultCode.RESOURCE_NOT_FOUND.code)


@router.post(
    "", response_model=Result[AlgorithmIdVO], summary="新增算法"
)
async def create_algorithm(
    body: AlgorithmForm,
    db: AsyncSession = Depends(get_db),
):
    algorithm_id = await AlgorithmService.create_algorithm(db, body.model_dump(exclude_none=True))
    return success(AlgorithmIdVO(id=algorithm_id), msg="算法创建成功")


@router.put("/{algorithm_id}", response_model=Result[None], summary="修改算法")
async def update_algorithm(
    algorithm_id: int,
    body: AlgorithmForm,
    db: AsyncSession = Depends(get_db),
):
    await AlgorithmService.update_algorithm(db, algorithm_id, body.model_dump(exclude_none=True))
    return success(msg="算法更新成功")


# ── 状态机 ──────────────────────────────────────

@router.put("/{algorithm_id}/status", response_model=Result[None], summary="修改算法状态")
async def update_algorithm_status(
    algorithm_id: int,
    body: AlgorithmStatusForm,
    db: AsyncSession = Depends(get_db),
):
    """
    修改算法状态（状态机流转）

    状态定义: 0=草稿 1=测试中 2=待审核 3=已发布 4=已停用 5=已归档

    合法流转:
    - 草稿 → 测试中
    - 测试中 → 待审核
    - 待审核 → 已发布 / 测试中（驳回）
    - 已发布 → 已停用
    - 已停用 → 已发布 / 已归档
    """
    await AlgorithmService.update_status(db, algorithm_id, body.status)
    return success(msg="算法状态更新成功")


# ── 审核 ──────────────────────────────────────

@router.put("/{algorithm_id}/audit", response_model=Result[None], summary="审核算法")
async def audit_algorithm(
    algorithm_id: int,
    body: AlgorithmAuditForm,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    审核算法（通过/驳回）

    - passed=true: 通过，状态变为已发布
    - passed=false: 驳回，必须填 remark，状态回到测试中
    """
    await AlgorithmService.audit_algorithm(
        db=db,
        algorithm_id=algorithm_id,
        audit_by=user.id,
        passed=body.passed,
        remark=body.remark,
    )
    return success(msg="算法审核完成")


# ── 版本控制 ──────────────────────────────────────

@router.post("/{algorithm_id}/version", response_model=Result[AlgorithmIdVO], summary="新增版本")
async def create_version(
    algorithm_id: int,
    body: AlgorithmVersionForm,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    新增算法版本

    - 校验版本号唯一（vX.Y.Z 格式）
    - 将当前版本归档到版本历史表
    - 更新算法主表
    - 失效该算法的预测缓存
    """
    from app.service.prediction_service import prediction_service
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
    await prediction_service.invalidate_cache(algorithm_id)
    return success(AlgorithmIdVO(id=algorithm_id), msg="版本创建成功")


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
    body: AlgorithmRollbackForm,
    db: AsyncSession = Depends(get_db),
):
    """
    回滚到指定版本

    - 仅已停用/已发布状态的算法可回滚
    - 将当前版本归档，应用目标版本
    """
    await AlgorithmService.rollback_version(db, algorithm_id, body.versionId)
    return success(msg="版本回滚成功")


# ── 删除 ──────────────────────────────────────

@router.delete(
    "", response_model=Result[AlgorithmDeleteResultVO], summary="批量删除算法"
)
async def delete_algorithms(
    ids: str = Query(..., description="算法ID，多个以逗号分隔"),
    db: AsyncSession = Depends(get_db),
):
    algorithm_ids = [int(i) for i in ids.split(",")]
    count = await AlgorithmService.delete_algorithms(db, algorithm_ids)
    return success(AlgorithmDeleteResultVO(count=count), msg="算法删除成功")


@router.delete("/{algorithm_id}", response_model=Result[AlgorithmDeleteResultVO], summary="删除单个算法")
async def delete_algorithm_single(
    algorithm_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    删除单个算法（含子算法）

    仅草稿/已停用状态的算法可删除。
    """
    count = await AlgorithmService.delete_algorithm_single(db, algorithm_id)
    return success(AlgorithmDeleteResultVO(count=count), msg="算法删除成功")


# ── 导入/导出 ──────────────────────────────────────

@router.get("/{algorithm_id}/_export", summary="导出单个算法（同步）")
async def export_algorithm(
    algorithm_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    同步导出单个算法为 ZIP

    ZIP 结构:
    - algorithm.json (算法元数据)
    - model/ (模型文件目录)
    """
    zip_bytes = await AlgorithmService.export_algorithm(db, algorithm_id)
    from datetime import datetime
    filename = f"algorithm_{algorithm_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
    return StreamingResponse(
        iter([zip_bytes]),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post(
    "/_import/validate",
    response_model=Result[AlgorithmImportResultVO],
    summary="校验导入包",
)
async def validate_import_package(
    file: UploadFile = File(..., description="算法包 ZIP 文件"),
):
    """校验算法导入包格式，不写入数据库"""
    file_bytes = await file.read()
    result = await AlgorithmService.validate_import_package(file_bytes)
    return success(AlgorithmImportResultVO(
        success=result["valid"],
        message=result["message"],
    ))


@router.post(
    "/_import",
    response_model=Result[AlgorithmIdVO],
    summary="导入算法包",
)
async def import_algorithm(
    file: UploadFile = File(..., description="算法包 ZIP 文件"),
    db: AsyncSession = Depends(get_db),
):
    """
    导入算法包

    - 校验包格式（algorithm.json + model/ 目录）
    - 名称唯一性校验
    - 创建算法记录（状态为草稿）
    - 解压模型文件到本地
    """
    file_bytes = await file.read()
    algorithm_id = await AlgorithmService.import_algorithm(db, file_bytes)
    return success(AlgorithmIdVO(id=algorithm_id), msg="算法导入成功")


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
    response_model=Result[AlgorithmMonitorStatsVO],
    summary="算法监控统计报表",
)
async def get_monitor_stats(
    algorithm_id: int,
    db: AsyncSession = Depends(get_db),
):
    """获取算法监控统计报表（时间序列 + 汇总）"""
    data = await AlgorithmService.get_monitor_stats_report(db, algorithm_id)
    return success(data)
