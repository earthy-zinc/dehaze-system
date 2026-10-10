"""Prometheus 指标：/metrics 供部署设施抓取，业务类指标留编排侧。

指标名不带服务前缀，服务来源由 Prometheus job 的 labels.service 区分（三服务统一）。
"""

import contextlib
import time
from collections.abc import Iterator

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

# endpoint 取路由路径（不含 query）：控制标签基数，避免查询串炸开时间序列
HTTP_REQUESTS = Counter("dehaze_http_requests_total", "HTTP 请求数", ["endpoint", "status"])
HTTP_DURATION = Histogram(
    "dehaze_http_request_duration_seconds",
    "HTTP 请求处理耗时（秒）",
    ["endpoint"],
    buckets=(0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0, 30.0, 60.0),
)

# 推理指标：type=chat/embedding/rerank；重排按请求计（内部逐对打分），向量按请求计（内部逐条）
INFERENCE_REQUESTS = Counter("dehaze_inference_requests_total", "推理请求数", ["type", "status"])
INFERENCE_DURATION = Histogram(
    "dehaze_inference_duration_seconds",
    "推理耗时（秒）",
    ["type"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0, 30.0, 60.0),
)


def record_inference(model_type: str, duration_seconds: float, status: str = "success") -> None:
    INFERENCE_REQUESTS.labels(type=model_type, status=status).inc()
    INFERENCE_DURATION.labels(type=model_type).observe(duration_seconds)


@contextlib.contextmanager
def observe_inference(model_type: str) -> Iterator[None]:
    """记录一次推理的类型/耗时/成败（异常计 error 后原样抛出）"""
    started = time.perf_counter()
    try:
        yield
    except Exception:
        record_inference(model_type, time.perf_counter() - started, "error")
        raise
    record_inference(model_type, time.perf_counter() - started)


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
