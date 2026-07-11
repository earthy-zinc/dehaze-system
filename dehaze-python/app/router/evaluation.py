"""
评估 API 路由 —— 去雾效果评估

POST /api/v1/evaluation  → 执行效果评估（PSNR/SSIM/LPIPS/NIQE）
"""

import logging
import time

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.result import Result, success, error
from app.core.code import ResultCode
from app.dependencies.auth import get_current_user, UserContext
from app.service.prediction_service import prediction_service  # 复用图片下载
from algorithm.metrics import calculate as calculate_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/evaluation", tags=["评估"])


class EvaluationRequest(BaseModel):
    """评估请求"""
    algorithmId: int = Field(description="算法ID")
    predUrl: str = Field(description="预测结果图片URL")
    gtUrl: str = Field(description="Ground Truth 参考图片URL")


class EvaluationResponse(BaseModel):
    """评估响应"""
    logId: int | None = Field(default=None, description="评估日志ID")
    metrics: dict[str, float] = Field(description="评估指标 {psnr, ssim, lpips, niqe, entropy}")
    time: int = Field(default=0, description="处理时间(毫秒)")


@router.post("", response_model=Result[EvaluationResponse])
async def evaluate(
    body: EvaluationRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    执行效果评估

    对比预测结果与参考图像，计算 PSNR/SSIM/LPIPS/NIQE 等多维指标
    """
    logger.info(f"评估请求: user={user.username}, algorithmId={body.algorithmId}")

    start = time.time()
    try:
        # 下载预测图和参考图
        pred_bytes = await prediction_service._download_image(body.predUrl)
        gt_bytes = await prediction_service._download_image(body.gtUrl)

        # 调用评估（返回列表 [{metric_name, value, ...}, ...]）
        pred_bytes.seek(0)
        gt_bytes.seek(0)
        metrics_list = calculate_metrics(pred_bytes, gt_bytes)

        # 转换为 { "psnr": 35.0, "ssim": 0.92, ... } 格式
        metrics = {m["metric_name"]: m["value"] for m in metrics_list}

        elapsed = int((time.time() - start) * 1000)
        logger.info(f"评估完成: algorithmId={body.algorithmId}, time={elapsed}ms, metrics={metrics}")

        return success(EvaluationResponse(
            logId=None,
            metrics=metrics,
            time=elapsed,
        ))

    except FileNotFoundError as e:
        return error(f"图片文件不存在: {e}", ResultCode.RESOURCE_NOT_FOUND.code)
    except Exception as e:
        logger.exception(f"评估失败: {e}")
        return error(f"评估执行失败: {e}", ResultCode.SYSTEM_EXECUTION_ERROR.code)
