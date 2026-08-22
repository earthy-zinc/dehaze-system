"""
评估 API 路由 —— 去雾效果评估（异步任务模式）

POST /api/v1/evaluation          → 提交评估任务，立即返回 {logId, status: "processing"}
GET  /api/v1/evaluation/logs     → 评估日志列表
GET  /api/v1/evaluation/metrics  → 当前用户评估指标历史
GET  /api/v1/evaluation/{taskId} → 查询评估任务状态，根据 status 返回不同字段
"""

import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.models.enum.log_status import LogStatus
from app.models.schema.common import PageResult
from app.service.evaluation_service import evaluation_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/evaluation", tags=["评估"], dependencies=[Depends(get_current_user)]
)


class EvaluationRequest(BaseModel):
    """评估请求

    与 Java 端 PythonAlgorithmClient.evaluate 的请求体一致：
    Java 已在 SysEvalLogServiceImpl 中将 predFileId/gtFileId 解析为 URL 字符串，
    始终以 predUrl/gtUrl 字符串形式转发给 Python，不再透传文件ID。
    """

    algorithmId: int = Field(description="算法ID")
    predUrl: str = Field(description="预测结果图片URL")
    gtUrl: str = Field(description="Ground Truth参考图片URL")
    params: str | None = Field(default=None, description="评估参数(JSON)")


class EvaluationResponse(BaseModel):
    """评估响应：POST 返回 logId+status；GET 根据 status 返回不同字段"""

    logId: int | None = Field(default=None, description="评估日志ID")
    status: int = Field(description="任务状态(1:处理中;2:已完成;3:失败)")
    metrics: dict[str, float] | None = Field(
        default=None, description="评估指标（completed 时返回）"
    )
    time: int = Field(default=0, description="处理时间(毫秒)")
    errorMessage: str | None = Field(default=None, description="失败错误信息（failed 时返回）")


@router.post("", response_model=Result[EvaluationResponse])
async def evaluate(
    body: EvaluationRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    提交效果评估任务（异步）

    立即返回 logId + status=processing，需通过 GET /{taskId} 轮询结果
    """
    result = await evaluation_service.evaluate(
        algorithm_id=body.algorithmId,
        pred_url=body.predUrl,
        gt_url=body.gtUrl,
        user_id=user.id,
        skip_quota_check=user.is_m2m,
    )
    return success(
        EvaluationResponse(
            logId=result.get("logId"),
            status=result.get("status", LogStatus.PROCESSING.value),
        )
    )


class EvaluationLogVO(BaseModel):
    """评估日志VO"""

    id: int = Field(description="日志ID")
    algorithmId: int = Field(
        validation_alias="algorithm_id", serialization_alias="algorithmId", description="算法ID"
    )
    predMd5: str | None = Field(
        default=None,
        validation_alias="pred_md5",
        serialization_alias="predMd5",
        description="预测图MD5",
    )
    predUrl: str | None = Field(
        default=None,
        validation_alias="pred_url",
        serialization_alias="predUrl",
        description="预测图URL",
    )
    gtMd5: str | None = Field(
        default=None, validation_alias="gt_md5", serialization_alias="gtMd5", description="GT图MD5"
    )
    gtUrl: str | None = Field(
        default=None, validation_alias="gt_url", serialization_alias="gtUrl", description="GT图URL"
    )
    status: int | None = Field(default=None, description="任务状态(1:处理中;2:已完成;3:失败)")
    errorMessage: str | None = Field(
        default=None,
        validation_alias="error_message",
        serialization_alias="errorMessage",
        description="失败错误信息",
    )
    time: int | None = Field(default=None, description="评估耗时(秒)")
    result: dict | None = Field(default=None, description="评估指标 JSON")
    createTime: datetime | None = Field(
        default=None,
        validation_alias="create_time",
        serialization_alias="createTime",
        description="创建时间",
    )

    model_config = {"populate_by_name": True}


def _to_log_dict(log) -> dict:
    """将评估日志实体组装为 VO dict（result 兼容 dict/str 两种存储形态）。"""
    return {
        "id": log.id,
        "algorithm_id": log.algorithm_id,
        "pred_md5": log.pred_md5,
        "pred_url": log.pred_url,
        "gt_md5": log.gt_md5,
        "gt_url": log.gt_url,
        "status": log.status,
        "error_message": log.error_message,
        "time": log.time,
        "result": log.result
        if isinstance(log.result, dict)
        else (json.loads(log.result) if isinstance(log.result, str) and log.result else None),
        "create_time": log.create_time,
    }


@router.get("/logs", response_model=Result[PageResult[EvaluationLogVO]], summary="评估日志列表")
async def list_evaluation_logs(
    algorithmId: int | None = Query(default=None, description="算法ID筛选"),
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=10, ge=1, le=100, description="每页数量"),
    db: AsyncSession = Depends(get_db),
):
    """分页查询评估日志"""
    logs, total = await evaluation_service.list_logs(
        db,
        algorithm_id=algorithmId,
        page=pageNum,
        size=pageSize,
    )
    return success(PageResult(list=[_to_log_dict(log) for log in logs], total=total))


@router.get("/metrics", response_model=Result[PageResult[EvaluationLogVO]], summary="评估指标历史")
async def get_evaluation_metrics(
    algorithmId: int | None = Query(default=None, description="算法ID筛选"),
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=10, ge=1, le=100, description="每页数量"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    """获取当前用户的历史评估记录（仅返回已完成的评估）"""
    logs, total = await evaluation_service.list_completed_metrics(
        db=db,
        user_id=user.id,
        algorithm_id=algorithmId,
        page=pageNum,
        size=pageSize,
    )
    return success(PageResult(list=[_to_log_dict(log) for log in logs], total=total))


@router.get("/{task_id}", response_model=Result[EvaluationResponse], summary="查询评估任务状态")
async def get_evaluation_task(
    task_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    查询评估任务状态（通过日志ID查询）

    根据 status 返回不同字段：
    - processing: 仅返回 logId + status
    - completed: 返回完整结果（metrics、time）
    - failed: 返回 errorMessage + time
    """
    log = await evaluation_service.get_log(db, task_id)

    resp = EvaluationResponse(logId=log.id, status=log.status)
    if log.status == LogStatus.COMPLETED.value:
        if isinstance(log.result, str) and log.result:
            try:
                resp.metrics = json.loads(log.result)
            except json.JSONDecodeError:
                resp.metrics = None
        elif isinstance(log.result, dict):
            resp.metrics = log.result
        resp.time = log.time or 0
    elif log.status == LogStatus.FAILED.value:
        resp.errorMessage = log.error_message
        resp.time = log.time or 0
    return success(resp)
