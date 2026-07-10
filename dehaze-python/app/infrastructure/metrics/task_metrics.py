import logging
import time
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

TASK_QUEUE_DEPTH = Gauge(
    "dehaze_task_queue_depth",
    "Current number of tasks waiting in queue",
    ["task_type"],
)

TASK_PROCESSING_TOTAL = Counter(
    "dehaze_task_processing_total",
    "Total number of tasks processed",
    ["task_type", "status"],
)

TASK_PROCESSING_TIME = Histogram(
    "dehaze_task_processing_time_seconds",
    "Task processing time in seconds",
    ["task_type", "status"],
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0),
)

TASK_IN_PROGRESS = Gauge(
    "dehaze_task_in_progress",
    "Number of tasks currently being processed",
    ["task_type"],
)


def update_task_queue_depth(task_type: str, depth: int) -> None:
    """
    更新任务队列深度

    Args:
        task_type: 任务类型 (export, inference, etc.)
        depth: 当前队列深度
    """
    TASK_QUEUE_DEPTH.labels(task_type=task_type).set(depth)


def increment_task_in_progress(task_type: str) -> None:
    TASK_IN_PROGRESS.labels(task_type=task_type).inc()


def decrement_task_in_progress(task_type: str) -> None:
    TASK_IN_PROGRESS.labels(task_type=task_type).dec()


def record_task_completion(
    task_type: str,
    status: str,
    duration_seconds: float,
) -> None:
    """
    记录任务完成情况

    Args:
        task_type: 任务类型
        status: 任务状态 (success, failed, cancelled)
        duration_seconds: 任务处理耗时（秒）
    """
    TASK_PROCESSING_TOTAL.labels(task_type=task_type, status=status).inc()
    TASK_PROCESSING_TIME.labels(task_type=task_type, status=status).observe(duration_seconds)
    decrement_task_in_progress(task_type)


class TaskMetricsContext:
    """
    任务指标上下文管理器

    用法:
        async with TaskMetricsContext("export") as ctx:
            # 执行任务
            ...
            ctx.set_status("success")

        # 自动记录耗时和状态
    """

    def __init__(self, task_type: str):
        self.task_type = task_type
        self._status: str = "success"
        self._start_time: Optional[float] = None

    def set_status(self, status: str) -> None:
        self._status = status

    async def __aenter__(self) -> "TaskMetricsContext":
        self._start_time = time.monotonic()
        increment_task_in_progress(self.task_type)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        duration = time.monotonic() - (self._start_time or 0)

        if exc_type is not None:
            self._status = "failed"

        record_task_completion(
            task_type=self.task_type,
            status=self._status,
            duration_seconds=duration,
        )
        return False  # 不抑制异常
