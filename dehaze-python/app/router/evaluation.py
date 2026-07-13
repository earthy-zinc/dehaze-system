"""
评估 API 路由 —— 去雾效果评估

POST /api/v1/evaluation          → 执行效果评估（PSNR/SSIM/LPIPS/NIQE/Entropy）
GET  /api/v1/evaluation/logs     → 评估日志列表
GET  /api/v1/evaluation/{taskId} → 查询评估任务状态（通过日志ID）
"""

import json
import logging
import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success, error
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.auth import get_current_user, UserContext
from app.database import get_db
from app.models.entity.sys_log import SysEvalLog
from app.models.schema.common import PageResult
from app.repository.pred_eval_log_repository import eval_log_repository
from app.service.prediction_service import prediction_service
from algorithm.metrics import calculate as calculate_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/evaluation", tags=["评估"],
                   dependencies=[Depends(get_current_user)])

# 评估合格阈值（文档要求）
QUALIFIED_THRESHOLDS = {
    "psnr": 30.0,   # ≥ 30 dB
    "ssim": 0.8,    # ≥ 0.8
    "lpips": 0.3,   # ≤ 0.3
    "niqe": 5.0,    # ≤ 5.0
}


def _is_qualified(metrics: dict[str, float]) -> bool:
    """基于阈值判定是否合格"""
    if not metrics:
        return False
    psnr = metrics.get("psnr", 0)
    ssim = metrics.get("ssim", 0)
    lpips = metrics.get("lpips", 1.0)
    niqe = metrics.get("niqe", 99.0)
    return (psnr >= QUALIFIED_THRESHOLDS["psnr"]
            and ssim >= QUALIFIED_THRESHOLDS["ssim"]
            and lpips <= QUALIFIED_THRESHOLDS["lpips"]
            and niqe <= QUALIFIED_THRESHOLDS["niqe"])


class EvaluationRequest(BaseModel):
    """评估请求"""
    algorithmId: int = Field(description="算法ID")
    predFileId: Optional[int] = Field(default=None, description="预测结果文件ID")
    gtFileId: Optional[int] = Field(default=None, description="Ground Truth参考图片文件ID")
    params: Optional[str] = Field(default=None, description="评估参数(JSON)")


class EvaluationResponse(BaseModel):
    """评估响应"""
    logId: Optional[int] = Field(default=None, description="评估日志ID")
    metrics: dict[str, float] = Field(description="评估指标 {psnr, ssim, lpips, niqe, entropy}")
    qualified: bool = Field(default=False, description="是否合格（基于 PSNR≥30/SSIM≥0.8/LPIPS≤0.3/NIQE≤5.0）")
    time: int = Field(default=0, description="处理时间(毫秒)")


def _resolve_file_url(file_id: Optional[int], file_type: str) -> str:
    """根据文件ID构造下载URL（与Java resolveFileUrl一致）"""
    if file_id is not None:
        return f"/api/v1/files/download/{file_id}"
    label = "预测" if file_type == "pred" else "参考"
    raise BusinessException(f"缺少{label}图片")


@router.post("", response_model=Result[EvaluationResponse])
async def evaluate(
    body: EvaluationRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    执行效果评估

    对比预测结果与参考图像，计算 PSNR/SSIM/LPIPS/NIQE/Entropy 等多维指标，
    并基于阈值判定是否合格。
    """
    logger.info(f"评估请求: user={user.username}, algorithmId={body.algorithmId}")

    # 1. 校验算法存在（与Java一致）
    await prediction_service._get_algorithm(body.algorithmId)

    # 2. 解析文件URL
    pred_url = _resolve_file_url(body.predFileId, "pred")
    gt_url = _resolve_file_url(body.gtFileId, "gt")

    start = time.time()
    try:
        # 下载预测图和参考图
        pred_bytes = await prediction_service._download_image(pred_url)
        gt_bytes = await prediction_service._download_image(gt_url)

        # 计算图片 MD5
        from app.utils.file import calculate_bytes_md5
        pred_md5 = calculate_bytes_md5(pred_bytes)
        gt_md5 = calculate_bytes_md5(gt_bytes)

        # 调用评估
        pred_bytes.seek(0)
        gt_bytes.seek(0)
        metrics_list = calculate_metrics(pred_bytes, gt_bytes)

        # 转换为 { "psnr": 35.0, "ssim": 0.92, ... } 格式
        metrics = {m["metric_name"]: m["value"] for m in metrics_list}

        elapsed = int((time.time() - start) * 1000)
        qualified = _is_qualified(metrics)
        logger.info(f"评估完成: algorithmId={body.algorithmId}, time={elapsed}ms, "
                    f"qualified={qualified}, metrics={metrics}")

        # 写入评估日志
        log_id = await _write_eval_log(
            algorithm_id=body.algorithmId,
            pred_md5=pred_md5,
            pred_url=pred_url,
            gt_md5=gt_md5,
            gt_url=gt_url,
            result=metrics,
            time_ms=elapsed,
        )

        return success(EvaluationResponse(
            logId=log_id,
            metrics=metrics,
            qualified=qualified,
            time=elapsed,
        ))

    except BusinessException:
        raise
    except FileNotFoundError as e:
        return error(f"图片文件不存在: {e}", ResultCode.RESOURCE_NOT_FOUND.code)
    except Exception as e:
        logger.exception(f"评估失败: {e}")
        return error(f"评估执行失败: {e}", ResultCode.SYSTEM_EXECUTION_ERROR.code)


async def _write_eval_log(
    algorithm_id: int,
    pred_md5: str,
    pred_url: str,
    gt_md5: str,
    gt_url: str,
    result: dict,
    time_ms: int,
) -> Optional[int]:
    """写入评估日志"""
    from app.database import async_session_factory
    try:
        async with async_session_factory() as db:
            log = await eval_log_repository.create_log(
                db=db,
                algorithm_id=algorithm_id,
                pred_md5=pred_md5,
                pred_url=pred_url,
                gt_md5=gt_md5,
                gt_url=gt_url,
                result=result,
                time_ms=time_ms,
            )
            await db.commit()
            return log.id
    except Exception as e:
        logger.warning(f"写入评估日志失败: {e}")
        return None


class EvaluationLogVO(BaseModel):
    """评估日志VO"""
    id: int = Field(description="日志ID")
    algorithmId: int = Field(validation_alias="algorithm_id", serialization_alias="algorithmId", description="算法ID")
    predMd5: Optional[str] = Field(default=None, validation_alias="pred_md5", serialization_alias="predMd5", description="预测图MD5")
    predUrl: Optional[str] = Field(default=None, validation_alias="pred_url", serialization_alias="predUrl", description="预测图URL")
    gtMd5: Optional[str] = Field(default=None, validation_alias="gt_md5", serialization_alias="gtMd5", description="GT图MD5")
    gtUrl: Optional[str] = Field(default=None, validation_alias="gt_url", serialization_alias="gtUrl", description="GT图URL")
    time: Optional[int] = Field(default=None, description="评估耗时(秒)")
    result: Optional[dict] = Field(default=None, description="评估指标 JSON")
    createTime: Optional[datetime] = Field(default=None, validation_alias="create_time", serialization_alias="createTime", description="创建时间")

    model_config = {"populate_by_name": True}


@router.get("/logs", response_model=Result[PageResult[EvaluationLogVO]], summary="评估日志列表")
async def list_evaluation_logs(
    algorithmId: Optional[int] = Query(default=None, description="算法ID筛选"),
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=10, ge=1, le=100, description="每页数量"),
    db: AsyncSession = Depends(get_db),
):
    """分页查询评估日志"""
    logs, total = await eval_log_repository.get_paginated(
        db=db,
        algorithm_id=algorithmId,
        page=pageNum,
        size=pageSize,
    )
    # result 字段可能是 JSON 字符串，转换为 dict
    log_list = []
    for log in logs:
        log_dict = {
            "id": log.id,
            "algorithm_id": log.algorithm_id,
            "pred_md5": log.pred_md5,
            "pred_url": log.pred_url,
            "gt_md5": log.gt_md5,
            "gt_url": log.gt_url,
            "time": log.time,
            "result": log.result if isinstance(log.result, dict) else (
                json.loads(log.result) if isinstance(log.result, str) and log.result else None
            ),
            "create_time": log.create_time,
        }
        log_list.append(log_dict)
    return success(PageResult(list=log_list, total=total))


@router.get("/{task_id}", response_model=Result[EvaluationLogVO], summary="查询评估任务状态")
async def get_evaluation_task(
    task_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    查询评估任务状态（通过日志ID查询）

    文档中的 taskId 对应 sys_eval_log.id
    """
    stmt = select(SysEvalLog).where(SysEvalLog.id == task_id)
    result = await db.execute(stmt)
    log = result.scalar_one_or_none()
    if not log:
        return error("评估任务不存在", ResultCode.SYSTEM_EXECUTION_ERROR.code)
    return success(log)
