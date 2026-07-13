"""
预测 API 路由 —— 去雾处理核心入口

POST /api/v1/prediction          → 执行模型预测（去雾）
GET  /api/v1/prediction/logs     → 预测日志列表
GET  /api/v1/prediction/{taskId} → 查询预测任务状态（通过日志ID）
"""
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.core.result import Result, success, error
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.auth import get_current_user, UserContext
from app.database import get_db
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
    """预测响应"""
    logId: Optional[int] = Field(default=None, description="预测日志ID")
    resultUrl: str = Field(description="处理后的图片URL")
    resultThumbnailUrl: Optional[str] = Field(default=None, description="缩略图URL")
    time: int = Field(default=0, description="处理时间(毫秒)")
    fromCache: bool = Field(default=False, description="是否命中缓存")


@router.post("", response_model=Result[PredictionResponse])
async def predict(
    body: PredictionRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    执行模型预测（去雾处理）

    接收雾化图片URL或文件ID，调用指定算法进行去雾，返回处理后的图片URL。
    基于 (algorithmId, imageMd5) 的 Redis 缓存，缓存命中直接返回结果。
    """
    logger.info(f"预测请求: user={user.username}, algorithmId={body.algorithmId}")

    # 解析图片来源 URL（fileId 优先）
    image_url = body.imageUrl
    if body.fileId is not None:
        image_url = f"/api/v1/files/download/{body.fileId}"
    if not image_url:
        raise BusinessException("图片来源不能为空，请提供 fileId 或 imageUrl")

    params = None
    if body.params:
        import json
        try:
            params = json.loads(body.params)
        except json.JSONDecodeError:
            return error(f"参数格式错误: {body.params}", ResultCode.PARAM_ERROR.code)

    try:
        result = await prediction_service.predict(
            algorithm_id=body.algorithmId,
            image_url=image_url,
            params=params,
            user_id=user.id,
        )

        return success(PredictionResponse(
            logId=result.get("logId"),
            resultUrl=result["resultUrl"],
            resultThumbnailUrl=result.get("resultThumbnailUrl"),
            time=result.get("time", 0),
            fromCache=result.get("fromCache", False),
        ))

    except BusinessException:
        raise
    except FileNotFoundError as e:
        return error(f"图片文件不存在: {e}", ResultCode.RESOURCE_NOT_FOUND.code)
    except ValueError as e:
        return error(f"算法模块错误: {e}", ResultCode.SYSTEM_EXECUTION_ERROR.code)
    except Exception as e:
        logger.exception(f"预测失败: {e}")
        return error(f"预测执行失败: {e}", ResultCode.SYSTEM_EXECUTION_ERROR.code)


class PredictionLogVO(BaseModel):
    """预测日志VO"""
    id: int = Field(description="日志ID")
    algorithmId: int = Field(validation_alias="algorithm_id", serialization_alias="algorithmId", description="算法ID")
    originMd5: Optional[str] = Field(default=None, validation_alias="origin_md5", serialization_alias="originMd5", description="原图MD5")
    originUrl: Optional[str] = Field(default=None, validation_alias="origin_url", serialization_alias="originUrl", description="原图URL")
    predMd5: Optional[str] = Field(default=None, validation_alias="pred_md5", serialization_alias="predMd5", description="预测结果MD5")
    predUrl: Optional[str] = Field(default=None, validation_alias="pred_url", serialization_alias="predUrl", description="预测结果URL")
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


@router.get("/{task_id}", response_model=Result[PredictionLogVO], summary="查询预测任务状态")
async def get_prediction_task(
    task_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    查询预测任务状态（通过日志ID查询）

    文档中的 taskId 对应 sys_pred_log.id
    """
    from app.models.entity.sys_log import SysPredLog
    from sqlalchemy import select
    stmt = select(SysPredLog).where(SysPredLog.id == task_id)
    result = await db.execute(stmt)
    log = result.scalar_one_or_none()
    if not log:
        return error("预测任务不存在", ResultCode.SYSTEM_EXECUTION_ERROR.code)
    return success(log)
