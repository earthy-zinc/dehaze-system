"""
预测 API 路由 —— 去雾处理核心入口（异步任务模式）

POST /api/v1/prediction          → 提交预测任务，立即返回 {logId, status: "processing"}
GET  /api/v1/prediction/logs     → 预测日志列表
GET  /api/v1/prediction/{taskId} → 查询预测任务状态，根据 status 返回不同字段
"""
import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.core.result import Result, success
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.auth import get_current_user, UserContext
from app.database import get_db
from app.models.enum.log_status import LogStatus
from app.models.schema.common import PageResult
from app.repository.pred_eval_log_repository import pred_log_repository
from app.service.prediction_service import prediction_service
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/prediction", tags=["预测"],
                   dependencies=[Depends(get_current_user)])


class PredictionRequest(BaseModel):
    """预测请求"""
    algorithmId: int = Field(description="算法ID")
    fileId: Optional[int] = Field(default=None, description="原始图片文件ID")
    imageUrl: Optional[str] = Field(default=None, description="原始图片URL（与fileId二选一）")
    params: Optional[str] = Field(default=None, description="预测参数(JSON)")


class PredictionResponse(BaseModel):
    """预测响应：POST 返回 logId+status；GET 根据 status 返回不同字段"""
    logId: Optional[int] = Field(default=None, description="预测日志ID")
    status: int = Field(description="任务状态(1:处理中;2:已完成;3:失败)")
    resultUrl: Optional[str] = Field(default=None, description="处理后的图片URL（completed 时返回）")
    resultThumbnailUrl: Optional[str] = Field(default=None, description="缩略图URL（completed 时返回）")
    time: int = Field(default=0, description="处理时间(毫秒)")
    errorMessage: Optional[str] = Field(default=None, description="失败错误信息（failed 时返回）")


@router.post("", response_model=Result[PredictionResponse])
async def predict(
    body: PredictionRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    提交模型预测任务（异步）

    立即返回 logId + status：
    - 缓存命中：status=completed 且包含完整结果
    - 缓存未命中：status=processing，需通过 GET /{taskId} 轮询
    """
    logger.info(f"预测请求: user={user.username}, algorithmId={body.algorithmId}")

    image_url = body.imageUrl
    if body.fileId is None and not image_url:
        raise BusinessException(ResultCode.PARAM_IS_NULL, "图片来源不能为空，请提供 fileId 或 imageUrl")

    params = None
    if body.params:
        try:
            params = json.loads(body.params)
        except json.JSONDecodeError:
            raise BusinessException(ResultCode.PARAM_ERROR, f"参数格式错误: {body.params}")

    result = await prediction_service.predict(
        algorithm_id=body.algorithmId,
        image_url=image_url,
        params=params,
        user_id=user.id,
        file_id=body.fileId,
        skip_quota_check=user.is_m2m,
    )

    return success(PredictionResponse(
        logId=result.get("logId"),
        status=result.get("status", LogStatus.PROCESSING.value),
        resultUrl=result.get("resultUrl"),
        resultThumbnailUrl=result.get("resultThumbnailUrl"),
        time=result.get("time", 0),
    ))


class PredictionLogVO(BaseModel):
    """预测日志VO"""
    id: int = Field(description="日志ID")
    algorithmId: int = Field(validation_alias="algorithm_id", serialization_alias="algorithmId", description="算法ID")
    originMd5: Optional[str] = Field(default=None, validation_alias="origin_md5", serialization_alias="originMd5", description="原图MD5")
    originUrl: Optional[str] = Field(default=None, validation_alias="origin_url", serialization_alias="originUrl", description="原图URL")
    predMd5: Optional[str] = Field(default=None, validation_alias="pred_md5", serialization_alias="predMd5", description="预测结果MD5")
    predUrl: Optional[str] = Field(default=None, validation_alias="pred_url", serialization_alias="predUrl", description="预测结果URL")
    status: Optional[int] = Field(default=None, description="任务状态(1:处理中;2:已完成;3:失败)")
    errorMessage: Optional[str] = Field(default=None, validation_alias="error_message", serialization_alias="errorMessage", description="失败错误信息")
    time: Optional[int] = Field(default=None, description="推理耗时(秒)")
    createTime: Optional[datetime] = Field(default=None, validation_alias="create_time", serialization_alias="createTime", description="创建时间")

    model_config = {"populate_by_name": True}


@router.get("/logs", response_model=Result[PageResult[PredictionLogVO]], summary="预测日志列表")
async def list_prediction_logs(
    algorithmId: Optional[int] = Query(default=None, description="算法ID筛选"),
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=10, ge=1, le=100, description="每页数量"),
    db: AsyncSession = Depends(get_db),
):
    """分页查询预测日志"""
    logs, total = await pred_log_repository.get_paginated(
        db=db,
        algorithm_id=algorithmId,
        page=pageNum,
        size=pageSize,
    )
    return success(PageResult(list=logs, total=total))


@router.get("/{task_id}", response_model=Result[PredictionResponse], summary="查询预测任务状态")
async def get_prediction_task(
    task_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    查询预测任务状态（通过日志ID查询）

    根据 status 返回不同字段：
    - processing: 仅返回 logId + status
    - completed: 返回完整结果（resultUrl、time）
    - failed: 返回 errorMessage + time
    """
    log = await pred_log_repository.get_by_id(db, task_id)
    if not log:
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "预测任务不存在")

    resp = PredictionResponse(logId=log.id, status=log.status)
    if log.status == LogStatus.COMPLETED.value:
        resp.resultUrl = log.pred_url
        resp.time = log.time or 0
    elif log.status == LogStatus.FAILED.value:
        resp.errorMessage = log.error_message
        resp.time = log.time or 0
    return success(resp)
