import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.infrastructure.logging import set_request_context


class TraceMiddleware(BaseHTTPMiddleware):
    """TraceID 中间件：为每个请求生成或透传 TraceID，用于日志关联和分布式追踪"""

    async def dispatch(self, request: Request, call_next) -> Response:
        # 优先从上游 header 获取（Java 后端传递），否则自动生成
        trace_id = (
            request.headers.get("X-Trace-Id")
            or request.headers.get("X-Request-Id")
            or uuid.uuid4().hex
        )

        # 注入请求上下文（traceId/method/path/ip/userAgent），供 JsonFormatter 自动写入每条日志
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

        response = await call_next(request)

        response.headers["X-Trace-Id"] = trace_id

        return response
