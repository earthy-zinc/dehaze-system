"""请求入口中间件：注入 trace_id、落访问日志、打点 HTTP 指标（日志架构设计 §4.1、§5.1）。"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.logging import set_request_context
from app.metrics import HTTP_DURATION, HTTP_REQUESTS

# 固定 logger 名（对齐日志架构设计的 ACCESS 示例）：检索入口日志不依赖模块名
logger = logging.getLogger("app.request")


class TraceContextMiddleware(BaseHTTPMiddleware):
    """透传/生成 X-Trace-Id 并落一条 ACCESS 日志；非 http（WebSocket）直接放行。

    WebSocket 无 HTTP 响应可记账，其耗时由流式路由的推理指标覆盖。
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = request.headers.get("X-Trace-Id") or uuid.uuid4().hex
        set_request_context(trace_id=trace_id, method=request.method, path=request.url.path)
        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        response.headers["X-Trace-Id"] = trace_id
        endpoint = request.url.path
        HTTP_REQUESTS.labels(endpoint=endpoint, status=response.status_code).inc()
        HTTP_DURATION.labels(endpoint=endpoint).observe(duration_ms / 1000)
        fields = {"status": response.status_code, "duration_ms": duration_ms}
        if request.url.query:
            fields["query"] = request.url.query
        logger.info("ACCESS", extra=fields)
        return response
