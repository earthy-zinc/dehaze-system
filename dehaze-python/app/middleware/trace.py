import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.infrastructure.logging import _trace_id_var


class TraceMiddleware(BaseHTTPMiddleware):
    """TraceID 中间件：为每个请求生成或透传 TraceID，用于日志关联和分布式追踪"""

    async def dispatch(self, request: Request, call_next) -> Response:
        # 优先从上游 header 获取（Java 后端传递），否则自动生成
        trace_id = request.headers.get(
            "X-Trace-Id") or request.headers.get("X-Request-Id") or str(uuid.uuid4().hex[:16])

        _trace_id_var.set(trace_id)

        response = await call_next(request)

        response.headers["X-Trace-Id"] = trace_id

        return response
