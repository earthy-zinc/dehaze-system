"""
图像任务 API 路由 —— 去雾处理核心入口（提交-轮询任务模式，设计 §2.7）

POST /api/v1/images/edits                 → 提交任务，立即返回 {taskId, status, estimatedCredits}
GET  /api/v1/images/tasks/{taskId}        → 查询任务状态，按 status 返回不同字段
POST /api/v1/images/tasks/{taskId}/cancel → 取消任务（幂等）
GET  /api/v1/images/tasks                 → 历史任务分页

注：原 /prediction/batch（批量=多次提交，由 SDK 封装）与 /prediction/quota（配额展示改积分余额）
已退役。
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
from app.service.prediction.prediction_service import prediction_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/images", tags=["图像任务"], dependencies=[Depends(get_current_user)]
)


class ImageEditRequest(BaseModel):
    """图像编辑/去雾提交请求"""

    model: str | None = Field(
        default=None, description="图像模型标识（统一模型规范 slug）；为空取平台默认模型"
    )
    fileId: int | None = Field(default=None, description="原始图片文件ID")
    imageUrl: str | None = Field(default=None, description="原始图片URL（与fileId二选一）")
    params: str | None = Field(default=None, description="推理参数(JSON)")
    recommendedBy: int | None = Field(
        default=None, description="推荐来源：推荐记录ID（推荐管理模块返回的 recommendationId）"
    )


class ImageTaskResponse(BaseModel):
    """任务响应：提交返回 taskId+status(；查询根据 status 返回不同字段)"""

    taskId: int | None = Field(default=None, description="任务ID（sys_pred_log.id）")
    status: int = Field(description="任务状态(1:处理中;2:已完成;3:失败;4:已取消)")
    estimatedCredits: int | None = Field(default=None, description="预估消耗积分（提交时返回）")
    resultUrl: str | None = Field(default=None, description="处理后的图片URL（completed 时返回）")
    resultMd5: str | None = Field(
        default=None, description="处理结果图片MD5（completed 时返回，下游落库用）"
    )
    originMd5: str | None = Field(
        default=None, description="原始图片内容MD5（completed 时返回，下游落库用）"
    )
    resultThumbnailUrl: str | None = Field(
        default=None, description="缩略图URL（completed 时返回）"
    )
    time: int = Field(default=0, description="推理耗时(秒)")
    errorMessage: str | None = Field(default=None, description="失败错误信息（failed 时返回）")


class ImageTaskVO(BaseModel):
    """历史任务项 VO"""

    id: int = Field(description="任务ID")
    model: str = Field(
        validation_alias="model_id", serialization_alias="model", description="图像模型标识"
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
        description="结果MD5",
    )
    predUrl: str | None = Field(
        default=None,
        validation_alias="pred_url",
        serialization_alias="predUrl",
        description="结果URL",
    )
    status: int | None = Field(
        default=None, description="任务状态(1:处理中;2:已完成;3:失败;4:已取消)"
    )
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


@router.post("/edits", response_model=Result[ImageTaskResponse], summary="提交图像编辑任务")
async def submit_image_edit(
    body: ImageEditRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    提交图像编辑/去雾任务（异步）

    立即返回 taskId + status：
    - 缓存/拦截器命中：status=completed 且包含完整结果（不计费）
    - 未命中：status=processing，需通过 GET /tasks/{taskId} 轮询
    """
    logger.debug(f"图像任务提交: user={user.username}, model={body.model}")

    image_url = body.imageUrl or ""
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
        model=body.model,
        image_url=image_url,
        params=params,
        user_id=user.id,
        file_id=body.fileId,
        skip_billing=user.is_m2m,
        recommended_by=body.recommendedBy,
    )

    return success(
        ImageTaskResponse(
            taskId=result.get("taskId"),
            status=result.get("status", LogStatus.PROCESSING.value),
            estimatedCredits=result.get("estimatedCredits"),
            resultUrl=result.get("resultUrl"),
            resultMd5=result.get("resultMd5"),
            originMd5=result.get("originMd5"),
            resultThumbnailUrl=result.get("resultThumbnailUrl"),
            time=result.get("time", 0),
        )
    )


@router.get(
    "/tasks",
    response_model=Result[PageResult[ImageTaskVO]],
    summary="图像任务历史列表",
)
async def list_image_tasks(
    model: str | None = Query(default=None, description="图像模型标识筛选"),
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=10, ge=1, le=100, description="每页数量"),
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """分页查询当前用户的图像任务历史"""
    logs, total = await prediction_service.list_logs(
        db, user_id=user.id, model_id=model, page=pageNum, size=pageSize
    )
    return success(PageResult(list=logs, total=total))


@router.get(
    "/tasks/{task_id}",
    response_model=Result[ImageTaskResponse],
    summary="查询图像任务状态",
)
async def get_image_task(
    task_id: str,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    查询图像任务状态（通过任务ID查询）

    根据 status 返回不同字段：
    - processing: 仅返回 taskId + status
    - completed: 返回完整结果（resultUrl、time）
    - failed: 返回 errorMessage + time
    """
    try:
        tid = int(task_id)
    except (ValueError, TypeError):
        raise BusinessException(ResultCode.PARAM_ERROR, f"无效的任务ID: {task_id}") from None
    log = await prediction_service.get_log(db, tid)
    # 归属校验：仅任务本人可查询（含结果图 URL），他人任务与不存在任务同样返回 A0401 防枚举
    if log.create_by != user.id:
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "图像任务不存在")

    resp = ImageTaskResponse(taskId=log.id, status=log.status)
    if log.status == LogStatus.COMPLETED.value:
        resp.resultUrl = log.pred_url
        resp.resultMd5 = log.pred_md5
        resp.originMd5 = log.origin_md5
        resp.time = log.time or 0
    elif log.status == LogStatus.FAILED.value:
        resp.errorMessage = log.error_message
        resp.time = log.time or 0
    # CANCELLED(4)：仅返回 taskId + status，无结果字段
    return success(resp)


@router.post(
    "/tasks/{task_id}/cancel",
    response_model=Result[ImageTaskResponse],
    summary="取消图像任务",
)
async def cancel_image_task(
    task_id: str,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    取消图像任务（幂等）。

    - 仅"处理中(1)"任务可取消：终止推理、状态置为"已取消(4)"。
    - 已完成(2)/已失败(3)/已取消(4)任务返回当前状态。
    - 任务不存在返回 A0401。
    """
    try:
        tid = int(task_id)
    except (ValueError, TypeError):
        raise BusinessException(ResultCode.PARAM_ERROR, f"无效的任务ID: {task_id}") from None

    result = await prediction_service.cancel_task(db, tid, user.id)
    return success(ImageTaskResponse(taskId=result["taskId"], status=result["status"]))
