"""Prometheus 指标：/metrics 供部署设施抓取，业务类指标留编排侧。

指标名不带服务前缀，服务来源由 Prometheus job 的 labels.service 区分（三服务统一）。
"""

from __future__ import annotations

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
INFERENCE_DURATION = Histogram("dehaze_inference_duration_seconds", "引擎推理耗时", ["engine"])


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
