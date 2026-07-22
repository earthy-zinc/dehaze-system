"""
评估 API 路由 —— 去雾效果评估

POST /api/v1/evaluation          → 执行效果评估（PSNR/SSIM/LPIPS/NIQE/Entropy）
GET  /api/v1/evaluation/logs     → 评估日志列表
GET  /api/v1/evaluation/{taskId} → 查询评估任务状态（通过日志ID）
"""

import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.auth import get_current_user, UserContext
from app.database import get_db
from app.models.schema.common import PageResult
from app.repository.pred_eval_log_repository import eval_log_repository
from app.service.evaluation_service import evaluation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/evaluation", tags=["评估"],
                   dependencies=[Depends(get_current_user)])


class EvaluationRequest(BaseModel):
    """评估请求

    与 Java 端 PythonAlgorithmClient.evaluate 的请求体一致：
    Java 已在 SysEvalLogServiceImpl 中将 predFileId/gtFileId 解析为 URL 字符串，
    始终以 predUrl/gtUrl 字符串形式转发给 Python，不再透传文件ID。
    """
    algorithmId: int = Field(description="算法ID")
    predUrl: str = Field(description="预测结果图片URL")
    gtUrl: str = Field(description="Ground Truth参考图片URL")
    params: Optional[str] = Field(default=None, description="评估参数(JSON)")


class EvaluationResponse(BaseModel):
    """评估响应"""
    logId: Optional[int] = Field(default=None, description="评估日志ID")
    metrics: dict[str, float] = Field(description="评估指标 {psnr, ssim, lpips, niqe, entropy}")
    qualified: bool = Field(default=False, description="是否合格（基于 PSNR≥30/SSIM≥0.8/LPIPS≤0.3/NIQE≤5.0）")
    time: int = Field(default=0, description="处理时间(毫秒)")


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
    result = await evaluation_service.evaluate(
        algorithm_id=body.algorithmId,
        pred_url=body.predUrl,
        gt_url=body.gtUrl,
    )
    return success(EvaluationResponse(
        logId=result["logId"],
        metrics=result["metrics"],
        qualified=result["qualified"],
        time=result["time"],
    ))


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
    log = await eval_log_repository.get_by_id(db, task_id)
    if not log:
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "评估任务不存在")
    return success(log)
