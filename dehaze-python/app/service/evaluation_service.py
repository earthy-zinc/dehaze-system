"""
评估服务 —— 编排去雾效果评估流程

对比预测结果与参考图像，计算 PSNR/SSIM/LPIPS/NIQE/Entropy 等多维指标，
并基于阈值判定是否合格。
"""

import asyncio
import logging
import time
from typing import Optional

from app.core.exceptions import BusinessException
from app.database import async_session_factory
from app.repository.pred_eval_log_repository import eval_log_repository
from app.service.prediction_service import prediction_service
from app.utils.file import calculate_bytes_md5
from algorithm.metrics import calculate as calculate_metrics

logger = logging.getLogger(__name__)

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


def _resolve_file_url(file_id: Optional[int], file_type: str) -> str:
    """根据文件ID构造下载URL（与Java resolveFileUrl一致）"""
    if file_id is not None:
        return f"/api/v1/files/download/{file_id}"
    label = "预测" if file_type == "pred" else "参考"
    raise BusinessException(f"缺少{label}图片")


class EvaluationService:
    """去雾效果评估服务"""

    async def evaluate(
        self,
        algorithm_id: int,
        pred_file_id: Optional[int] = None,
        gt_file_id: Optional[int] = None,
    ) -> dict:
        """
        执行效果评估

        Returns:
            {
                "logId": Optional[int],
                "metrics": dict[str, float],
                "qualified": bool,
                "time": int (毫秒),
            }
        """
        logger.info("评估请求: algorithmId=%s", algorithm_id)

        # 1. 校验算法存在
        await prediction_service.get_algorithm(algorithm_id)

        # 2. 解析文件URL
        pred_url = _resolve_file_url(pred_file_id, "pred")
        gt_url = _resolve_file_url(gt_file_id, "gt")

        start = time.time()

        # 3. 并行下载预测图和参考图
        pred_bytes, gt_bytes = await asyncio.gather(
            prediction_service.download_image(pred_url),
            prediction_service.download_image(gt_url),
        )

        # 4. 计算图片 MD5（CPU 密集型，移至线程池）
        pred_md5, gt_md5 = await asyncio.gather(
            asyncio.to_thread(calculate_bytes_md5, pred_bytes),
            asyncio.to_thread(calculate_bytes_md5, gt_bytes),
        )

        # 5. 调用评估（图像处理 CPU 密集型，移至线程池避免阻塞事件循环）
        pred_bytes.seek(0)
        gt_bytes.seek(0)
        metrics_list = await asyncio.to_thread(
            calculate_metrics, pred_bytes, gt_bytes
        )

        # 转换为 { "psnr": 35.0, "ssim": 0.92, ... } 格式
        metrics = {m["metric_name"]: m["value"] for m in metrics_list}

        elapsed = int((time.time() - start) * 1000)
        qualified = _is_qualified(metrics)
        logger.info(
            "评估完成: algorithmId=%s, time=%sms, qualified=%s, metrics=%s",
            algorithm_id, elapsed, qualified, metrics,
        )

        # 6. 写入评估日志
        log_id = await self._write_eval_log(
            algorithm_id=algorithm_id,
            pred_md5=pred_md5,
            pred_url=pred_url,
            gt_md5=gt_md5,
            gt_url=gt_url,
            result=metrics,
            time_ms=elapsed,
        )

        return {
            "logId": log_id,
            "metrics": metrics,
            "qualified": qualified,
            "time": elapsed,
        }

    async def _write_eval_log(
        self,
        algorithm_id: int,
        pred_md5: str,
        pred_url: str,
        gt_md5: str,
        gt_url: str,
        result: dict,
        time_ms: int,
    ) -> Optional[int]:
        """写入评估日志"""
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
            logger.warning("写入评估日志失败: %s", e)
            return None


# 单例
evaluation_service = EvaluationService()
