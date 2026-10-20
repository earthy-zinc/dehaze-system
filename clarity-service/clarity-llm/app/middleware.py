"""请求入口中间件：注入请求上下文、落 ACCESS 日志、打点 HTTP 指标。"""

import logging
import time
import uuid

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.logging import set_request_context
from app.metrics import HTTP_DURATION, HTTP_REQUESTS

# 固定 logger 名：`python -m app.main` 下模块 __name__ 为 __main__，不利于检索
logger = logging.getLogger("app.request")

# 探针与指标抓取高频且无诊断价值：照常记指标，但不落 ACCESS 日志，避免淹没业务请求
_NO_ACCESS_LOG_PATHS = frozenset({"/health", "/ready", "/metrics"})


def _record_request(path: str, status: int, started: float) -> None:
    duration_ms = round((time.perf_counter() - started) * 1000, 3)
    HTTP_REQUESTS.labels(endpoint=path, status=status).inc()
    HTTP_DURATION.labels(endpoint=path).observe(duration_ms / 1000)
    if path not in _NO_ACCESS_LOG_PATHS:
        logger.info("ACCESS", extra={"status": status, "duration_ms": duration_ms})


class TraceContextMiddleware:
    """透传/生成 X-Trace-Id、回写响应头，并在响应头发出时落一条 ACCESS。

    纯 ASGI 实现而非 BaseHTTPMiddleware：后者在响应外层再包一层任务组，对流式响应的断连
    取消语义有干扰，而流式对话必须保持"客户端断连即停止生成"（与鉴权中间件同理）。
    鉴权失败的 401 同样被记账并落 ACCESS：本中间件在鉴权中间件外层。
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        trace_id = Headers(scope=scope).get("x-trace-id") or uuid.uuid4().hex
        set_request_context(trace_id=trace_id, method=scope["method"], path=path)
        started = time.perf_counter()

        async def send_with_access(message: Message) -> None:
            if message["type"] == "http.response.start":
                _record_request(path, message["status"], started)
                message = {
                    **message,
                    "headers": [*message.get("headers", ()), (b"x-trace-id", trace_id.encode())],
                }
            await send(message)

        await self.app(scope, receive, send_with_access)
