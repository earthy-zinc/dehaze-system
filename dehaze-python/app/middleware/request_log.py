"""请求级访问日志中间件

为每个进入的请求打印一条访问日志。method/path/trace_id 由 JsonFormatter 自动注入为
字段（由 TraceMiddleware 设置）；此处补充响应级字段 status/duration_ms 与请求 query，
均为独立 JSON 字段，便于按状态码 / 耗时 / 查询串检索。

请求 / 响应完整 body 不在此记录（涉及 PII / 密钥 / 体积 / 二进制），应交由
sys_operation_log 审计表（已设计 body/resp 列）按需落库。

须在 TraceMiddleware 之内执行（注册顺序上先于 TraceMiddleware 注册），此时请求上下文
已由 TraceMiddleware 注入；健康检查 / 指标 / 文档等噪声路径跳过，避免刷屏。
"""

import time
import logging

from starlette.middleware.base import BaseHTTPMiddleware

# 跳过访问日志的噪声路径（与 Prometheus skip_paths 对齐）
_SKIP_PATHS = frozenset({
    "/health", "/ready", "/metrics", "/docs", "/redoc", "/openapi.json",
})


class RequestLogMiddleware(BaseHTTPMiddleware):
    """每个请求打印一条访问日志。"""

    async def dispatch(self, request, call_next):
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        extra = {"status": response.status_code, "duration_ms": round(duration_ms, 1)}
        if request.url.query:
            extra["query"] = request.url.query
        logging.getLogger("app.request").info("ACCESS", extra=extra)
        return response
