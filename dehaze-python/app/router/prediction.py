"""
预测 API 路由 —— 去雾处理核心入口

POST /api/v1/prediction  → 执行模型预测（去雾）
"""
import time
from typing import Optional

from fastapi import APIRouter, Depends
from loguru import logger
from pydantic import BaseModel, Field

from app.core.result import Result, success, error
from app.core.code import ResultCode
from app.dependencies.auth import get_current_user, UserContext
from app.service.prediction_service import prediction_service

router = APIRouter(prefix="/api/v1/prediction", tags=["预测"])


class PredictionRequest(BaseModel):
    """预测请求"""
    algorithmId: int = Field(description="算法ID")
    imageUrl: str = Field(description="输入图片URL")
    params: Optional[str] = Field(default=None, description="预测参数(JSON)")


class PredictionResponse(BaseModel):
    """预测响应"""
    logId: Optional[int] = Field(default=None, description="预测日志ID")
    resultUrl: str = Field(description="处理后的图片URL")
    resultThumbnailUrl: Optional[str] = Field(default=None, description="缩略图URL")
    time: int = Field(default=0, description="处理时间(毫秒)")


@router.post("", response_model=Result[PredictionResponse])
async def predict(
    body: PredictionRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    执行模型预测（去雾处理）

    接收雾化图片URL，调用指定算法进行去雾，返回处理后的图片URL
    """
    logger.info(f"预测请求: user={user.username}, algorithmId={body.algorithmId}")

    params = None
    if body.params:
        try:
            import json
            params = json.loads(body.params)
        except json.JSONDecodeError:
            return error(f"参数格式错误: {body.params}", ResultCode.PARAM_ERROR.code)

    start = time.time()
    try:
        result = await prediction_service.predict(
            algorithm_id=body.algorithmId,
            image_url=body.imageUrl,
            params=params,
        )
        elapsed = int((time.time() - start) * 1000)
        logger.info(f"预测完成: algorithmId={body.algorithmId}, time={elapsed}ms")

        return success(PredictionResponse(
            logId=None,
            resultUrl=result["resultUrl"],
            resultThumbnailUrl=result.get("resultThumbnailUrl"),
            time=elapsed,
        ))

    except FileNotFoundError as e:
        return error(f"图片文件不存在: {e}", ResultCode.RESOURCE_NOT_FOUND.code)
    except ValueError as e:
        return error(f"算法模块错误: {e}", ResultCode.SYSTEM_RESOURCE_ERROR.code)
    except Exception as e:
        logger.exception(f"预测失败: {e}")
        return error(f"预测执行失败: {e}", ResultCode.SYSTEM_EXECUTION_ERROR.code)
