"""
预测 API 路由 —— 去雾处理核心入口（异步任务模式）

POST /api/v1/prediction          → 提交预测任务，立即返回 {logId, status: "processing"}
POST /api/v1/prediction/batch    → 批量处理
GET  /api/v1/prediction/quota    → 查询剩余处理次数
GET  /api/v1/prediction/logs     → 预测日志列表
GET  /api/v1/prediction/{taskId} → 查询预测任务状态，根据 status 返回不同字段
"""

import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.models.enum.log_status import LogStatus
from app.models.schema.common import PageResult
from app.models.schema.prediction import BatchPredictionItem
from app.service.prediction.prediction_service import prediction_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/prediction", tags=["预测"], dependencies=[Depends(get_current_user)]
)


class PredictionRequest(BaseModel):
    """预测请求"""

    algorithmId: int = Field(description="算法ID")
    fileId: int | None = Field(default=None, description="原始图片文件ID")
    imageUrl: str | None = Field(default=None, description="原始图片URL（与fileId二选一）")
    params: str | None = Field(default=None, description="预测参数(JSON)")
    recommendedBy: int | None = Field(
        default=None, description="推荐来源：推荐记录ID（推荐管理模块返回的 recommendationId）"
    )


class BatchPredictionRequest(BaseModel):
    """批量预测请求"""

    algorithmId: int = Field(description="算法ID")
    items: list[BatchPredictionItem] = Field(description="批量图片列表，每项含 fileId/imageUrl/params 等")
    recommendedBy: int | None = Field(default=None, description="推荐来源：推荐记录ID")


class PredictionResponse(BaseModel):
    """预测响应：POST 返回 logId+status；GET 根据 status 返回不同字段"""

    logId: int | None = Field(default=None, description="预测日志ID")
    status: int = Field(description="任务状态(1:处理中;2:已完成;3:失败)")
    resultUrl: str | None = Field(default=None, description="处理后的图片URL（completed 时返回）")
    resultThumbnailUrl: str | None = Field(
        default=None, description="缩略图URL（completed 时返回）"
    )
    time: int = Field(default=0, description="处理时间(毫秒)")
    errorMessage: str | None = Field(default=None, description="失败错误信息（failed 时返回）")


class BatchPredictionResult(BaseModel):
    """批量预测响应包装"""

    total: int = Field(description="本次提交的图片总数")
    results: list[PredictionResponse] = Field(description="每张图的预测结果列表")


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
    logger.debug(f"预测请求: user={user.username}, algorithmId={body.algorithmId}")

    image_url = body.imageUrl
    if body.fileId is None and not image_url:
        raise BusinessException(
            ResultCode.PARAM_IS_NULL, "图片来源不能为空，请提供 fileId 或 imageUrl"
        )

    params = None
    if body.params:
        try:
            params = json.loads(body.params)
        except json.JSONDecodeError:
            raise BusinessException(
                ResultCode.PARAM_ERROR, f"参数格式错误: {body.params}"
            ) from None

    result = await prediction_service.predict(
        algorithm_id=body.algorithmId,
        image_url=image_url,
        params=params,
        user_id=user.id,
        file_id=body.fileId,
        skip_quota_check=user.is_m2m,
    )

    return success(
        PredictionResponse(
            logId=result.get("logId"),
            status=result.get("status", LogStatus.PROCESSING.value),
            resultUrl=result.get("resultUrl"),
            resultThumbnailUrl=result.get("resultThumbnailUrl"),
            time=result.get("time", 0),
        )
    )


class PredictionLogVO(BaseModel):
    """预测日志VO"""

    id: int = Field(description="日志ID")
    algorithmId: int = Field(
        validation_alias="algorithm_id", serialization_alias="algorithmId", description="算法ID"
    )
    originMd5: str | None = Field(
        default=None,
        validation_alias="origin_md5",
        serialization_alias="originMd5",
        description="原图MD5",
    )
    originUrl: str | None = Field(
        default=None,
        validation_alias="origin_url",
        serialization_alias="originUrl",
        description="原图URL",
    )
    predMd5: str | None = Field(
        default=None,
        validation_alias="pred_md5",
        serialization_alias="predMd5",
        description="预测结果MD5",
    )
    predUrl: str | None = Field(
        default=None,
        validation_alias="pred_url",
        serialization_alias="predUrl",
        description="预测结果URL",
    )
    status: int | None = Field(default=None, description="任务状态(1:处理中;2:已完成;3:失败)")
    errorMessage: str | None = Field(
        default=None,
        validation_alias="error_message",
        serialization_alias="errorMessage",
        description="失败错误信息",
    )
    time: int | None = Field(default=None, description="推理耗时(秒)")
    createTime: datetime | None = Field(
        default=None,
        validation_alias="create_time",
        serialization_alias="createTime",
        description="创建时间",
    )

    model_config = {"populate_by_name": True}


@router.get("/logs", response_model=Result[PageResult[PredictionLogVO]], summary="预测日志列表")
async def list_prediction_logs(
    algorithmId: int | None = Query(default=None, description="算法ID筛选"),
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=10, ge=1, le=100, description="每页数量"),
    db: AsyncSession = Depends(get_db),
):
    """分页查询预测日志"""
    logs, total = await prediction_service.list_logs(
        db,
        algorithm_id=algorithmId,
        page=pageNum,
        size=pageSize,
    )
    return success(PageResult(list=logs, total=total))


class QuotaResponse(BaseModel):
    """配额查询响应"""

    total: int = Field(description="总配额")
    used: int = Field(description="本月已使用")
    remaining: int = Field(description="本月剩余")
    resetDate: str = Field(description="配额重置日期（月度配额为下月1日）")


@router.get("/quota", response_model=Result[QuotaResponse], summary="查询剩余处理次数")
async def get_quota(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """查询用户本月剩余去雾处理次数"""
    result = await prediction_service.get_quota(db, user.id)
    return success(QuotaResponse(**result))


@router.get("/{task_id}", response_model=Result[PredictionResponse], summary="查询预测任务状态")
async def get_prediction_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    查询预测任务状态（通过日志ID查询）

    根据 status 返回不同字段：
    - processing: 仅返回 logId + status
    - completed: 返回完整结果（resultUrl、time）
    - failed: 返回 errorMessage + time
    """
    try:
        tid = int(task_id)
    except (ValueError, TypeError):
        raise BusinessException(ResultCode.PARAM_ERROR, f"无效的任务ID: {task_id}") from None
    log = await prediction_service.get_log(db, tid)

    resp = PredictionResponse(logId=log.id, status=log.status)
    if log.status == LogStatus.COMPLETED.value:
        resp.resultUrl = log.pred_url
        resp.time = log.time or 0
    elif log.status == LogStatus.FAILED.value:
        resp.errorMessage = log.error_message
        resp.time = log.time or 0
    # CANCELLED(4)：仅返回 logId + status，无结果字段
    return success(resp)


@router.post(
    "/{task_id}/cancel",
    response_model=Result[PredictionResponse],
    summary="取消预测任务",
)
async def cancel_prediction_task(
    task_id: str,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    取消预测任务（幂等）。

    - 仅"处理中(1)"任务可取消：终止推理、回滚已扣减配额、状态置为"已取消(4)"。
    - 已完成(2)/已失败(3)/已取消(4)任务返回当前状态，不重复回滚配额。
    - 任务不存在返回 A0401。
    """
    try:
        tid = int(task_id)
    except (ValueError, TypeError):
        raise BusinessException(ResultCode.PARAM_ERROR, f"无效的任务ID: {task_id}") from None

    result = await prediction_service.cancel_task(db, tid, user.id)
    return success(PredictionResponse(logId=result["logId"], status=result["status"]))


@router.post("/batch", response_model=Result[BatchPredictionResult], summary="批量处理")
async def batch_predict(
    body: BatchPredictionRequest,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """批量提交去雾预测任务，最多20张（VIP差异）"""
    results = await prediction_service.batch_predict(
        algorithm_id=body.algorithmId,
        items=body.items,
        user_id=user.id,
        skip_quota_check=user.is_m2m,
    )
    return success(BatchPredictionResult(total=len(results), results=results))
