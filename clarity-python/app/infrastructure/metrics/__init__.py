"""
Prometheus 指标采集模块
提供缓存、任务队列等业务指标采集能力。
HTTP 请求指标由 starlette-exporter 中间件自动采集；推理与 GPU 指标随执行器
迁至 clarity-modal（服务侧 /metrics 暴露）。
"""

from app.infrastructure.metrics.cache_metrics import (
    CACHE_HITS_TOTAL,
    CACHE_LOADER_TOTAL,
    CACHE_MISSES_TOTAL,
    record_hit,
    record_loader,
    record_miss,
)
from app.infrastructure.metrics.task_metrics import (
    TASK_DURATION,
    TASK_QUEUE_DEPTH,
    TASK_TOTAL,
    update_task_queue_depth,
)

__all__ = [
    # 缓存指标
    "CACHE_HITS_TOTAL",
    "CACHE_LOADER_TOTAL",
    "CACHE_MISSES_TOTAL",
    # 任务指标
    "TASK_DURATION",
    "TASK_QUEUE_DEPTH",
    "TASK_TOTAL",
    "record_hit",
    "record_loader",
    "record_miss",
    "update_task_queue_depth",
]
