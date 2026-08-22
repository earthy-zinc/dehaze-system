"""推理指标 Prometheus 采集

提供 record_inference_metrics() 手动记录推理耗时、请求计数、图像尺寸等指标。
"""

import logging

from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# buckets: 100ms, 500ms, 1s, 2s, 5s, 10s, 30s, 1min, 2min, 5min
INFERENCE_DURATION = Histogram(
    "dehaze_inference_duration_seconds",
    "Inference request duration in seconds",
    ["algorithm", "status"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

INFERENCE_REQUESTS_TOTAL = Counter(
    "dehaze_inference_requests_total",
    "Total number of inference requests",
    ["algorithm", "status"],
)

INFERENCE_IMAGE_SIZE = Histogram(
    "dehaze_inference_image_size_pixels",
    "Inference input image size (width * height)",
    ["algorithm"],
    buckets=(256 * 256, 512 * 512, 1024 * 1024, 2048 * 2048, 4096 * 4096),
)

INFERENCE_BATCH_SIZE = Histogram(
    "dehaze_inference_batch_size",
    "Inference batch size",
    ["algorithm"],
    buckets=(1, 2, 4, 8, 16, 32),
)


def _record_inference(algorithm: str, status: str, duration: float) -> None:
    """记录推理指标（内部公共逻辑）"""
    INFERENCE_DURATION.labels(algorithm=algorithm, status=status).observe(duration)
    INFERENCE_REQUESTS_TOTAL.labels(algorithm=algorithm, status=status).inc()


def record_inference_metrics(
    algorithm: str,
    duration_seconds: float,
    status: str = "success",
    image_size: int | None = None,
    batch_size: int = 1,
) -> None:
    """
    记录推理指标

    Args:
        algorithm: 算法名称
        duration_seconds: 推理耗时（秒）
        status: 状态 (success/error)
        image_size: 图像尺寸（宽*高像素数）
        batch_size: 批处理大小
    """
    _record_inference(algorithm, status, duration_seconds)

    if image_size is not None:
        INFERENCE_IMAGE_SIZE.labels(algorithm=algorithm).observe(image_size)

    if batch_size > 0:
        INFERENCE_BATCH_SIZE.labels(algorithm=algorithm).observe(batch_size)
