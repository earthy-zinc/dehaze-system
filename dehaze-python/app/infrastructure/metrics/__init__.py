"""
Prometheus 指标采集模块

提供 GPU 利用率、推理耗时、任务队列等业务指标采集能力。
HTTP 请求指标由 starlette-exporter 中间件自动采集。
"""

from app.infrastructure.metrics.cache_metrics import (CACHE_HITS_TOTAL,
                                                     CACHE_MISSES_TOTAL,
                                                     CACHE_LOADER_TOTAL,
                                                     record_hit, record_miss,
                                                     record_loader)
from app.infrastructure.metrics.gpu_metrics import GPUMetricsCollector, \
    collect_gpu_metrics, get_gpu_metrics_collector
from app.infrastructure.metrics.inference_metrics import (
    INFERENCE_DURATION, INFERENCE_REQUESTS_TOTAL, record_inference_metrics)
from app.infrastructure.metrics.task_metrics import (TASK_PROCESSING_TIME,
                                                     TASK_PROCESSING_TOTAL,
                                                     TASK_QUEUE_DEPTH,
                                                     update_task_queue_depth)

__all__ = [
    # GPU 指标
    "GPUMetricsCollector",
    "collect_gpu_metrics",
    "get_gpu_metrics_collector",
    # 推理指标
    "INFERENCE_DURATION",
    "INFERENCE_REQUESTS_TOTAL",
    "record_inference_metrics",
    # 任务指标
    "TASK_QUEUE_DEPTH",
    "TASK_PROCESSING_TOTAL",
    "TASK_PROCESSING_TIME",
    "update_task_queue_depth",
    # 缓存指标
    "CACHE_HITS_TOTAL",
    "CACHE_MISSES_TOTAL",
    "CACHE_LOADER_TOTAL",
    "record_hit",
    "record_miss",
    "record_loader",
]
