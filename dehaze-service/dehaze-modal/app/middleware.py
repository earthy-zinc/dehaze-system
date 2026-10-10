"""请求入口中间件：注入请求上下文、落 ACCESS 日志、打点 HTTP 指标。"""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.logging import set_request_context
from app.metrics import HTTP_DURATION, HTTP_REQUESTS

# 固定 logger 名：`python -m app.main` 下模块 __name__ 为 __main__，不利于检索
logger = logging.getLogger("app.request")


class TraceContextMiddleware(BaseHTTPMiddleware):
    """透传/生成 X-Trace-Id、回写响应头，并落一条 ACCESS 日志。

    探针与 /metrics 同样计数：抓取流量本身就是需要观测的请求量。
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = (
            request.headers.get("X-Trace-Id")
            or request.headers.get("X-Request-Id")
            or uuid.uuid4().hex
        )
        forwarded = request.headers.get("x-forwarded-for", "")
        ip = (
            forwarded.split(",")[0].strip()
            if forwarded
            else (request.client.host if request.client else "")
        )
        set_request_context(
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
            ip=ip,
            user_agent=request.headers.get("user-agent", ""),
        )

        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        response.headers["X-Trace-Id"] = trace_id
        HTTP_REQUESTS.labels(endpoint=request.url.path, status=response.status_code).inc()
        HTTP_DURATION.labels(endpoint=request.url.path).observe(duration_ms / 1000)

        fields = {"status": response.status_code, "duration_ms": duration_ms}
        if request.url.query:
            fields["query"] = request.url.query
        logger.info("ACCESS", extra=fields)
        return response
