import asyncio
import contextlib
import functools
import logging
import time
from typing import Any, Awaitable, Callable, Optional, TypeVar, overload

from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

T = TypeVar("T")

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
    INFERENCE_DURATION.labels(
        algorithm=algorithm, status=status).observe(duration)
    INFERENCE_REQUESTS_TOTAL.labels(algorithm=algorithm, status=status).inc()


@contextlib.contextmanager
def _track_timing(algorithm: str):
    """推理计时上下文管理器，统一 async/sync 的指标记录逻辑"""
    start = time.monotonic()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        _record_inference(algorithm, status, time.monotonic() - start)


def track_inference(
    algorithm: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    推理耗时追踪装饰器

    用法:
        @track_inference(algorithm="AECRNet")
        async def run_inference(self, image: Image) -> Result:
            ...

    Args:
        algorithm: 算法名称
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with _track_timing(algorithm):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with _track_timing(algorithm):
                return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def record_inference_metrics(
    algorithm: str,
    duration_seconds: float,
    status: str = "success",
    image_size: Optional[int] = None,
    batch_size: int = 1,
) -> None:
    """
    手动记录推理指标

    用于无法使用装饰器的场景。

    Args:
        algorithm: 算法名称
        duration_seconds: 推理耗时（秒）
        status: 状态 (success/error)
        image_size: 图像尺寸（宽*高）
        batch_size: 批处理大小
    """
    _record_inference(algorithm, status, duration_seconds)

    if image_size is not None:
        INFERENCE_IMAGE_SIZE.labels(algorithm=algorithm).observe(image_size)

    if batch_size > 0:
        INFERENCE_BATCH_SIZE.labels(algorithm=algorithm).observe(batch_size)
