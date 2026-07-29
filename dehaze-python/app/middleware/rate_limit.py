"""限流中间件（纯 ASGI 实现）

基于 IP + 路径的固定窗口限流，使用 Redis INCR + EXPIRE 实现。
Key 格式：rate:limit:{path}:{ip}
默认限制：每个 IP 每分钟 60 次请求（可通过配置调整）。
Redis 不可用时降级放行，避免影响业务主流程。
"""
import json
import logging

from starlette.types import ASGIApp, Receive, Scope, Send

from app.config import settings
from app.core.code import ResultCode
from app.dependencies.redis import get_redis_client
from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback

logger = logging.getLogger(__name__)

_EXCLUDE_PATHS = {
    "/health",
    "/health/db",
    "/health/redis",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
}


async def _send_json_response(send: Send, status_code: int, content: dict):
    body = json.dumps(content, ensure_ascii=False).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": status_code,
        "headers": [
            (b"content-type", b"application/json; charset=utf-8"),
            (b"content-length", str(len(body)).encode("latin-1")),
        ],
    })
    await send({
        "type": "http.response.body",
        "body": body,
    })


class RateLimitMiddleware:
    """基于 IP + 路径的固定窗口限流中间件"""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        if not settings.RATE_LIMIT_ENABLED:
            return await self.app(scope, receive, send)

        path: str = scope["path"]
        if path in _EXCLUDE_PATHS:
            return await self.app(scope, receive, send)

        client = scope.get("client")
        if not client:
            return await self.app(scope, receive, send)

        ip = client[0]
        redis = await get_redis_client()
        if not redis:
            return await self.app(scope, receive, send)

        cache_key = f"rate:limit:{path}:{ip}"

        async def _check():
            count = await redis.incr(cache_key)
            if count == 1:
                await redis.expire(cache_key, settings.RATE_LIMIT_WINDOW_SECONDS)
            return count

        count = await redis_operation_with_fallback(
            operation=_check,
            default=0,
            operation_name=f"rate:limit:{path}",
        )

        if count and count > settings.RATE_LIMIT_MAX_REQUESTS:
            logger.warning(
                f"限流拦截: ip={ip}, path={path}, count={count}, "
                f"limit={settings.RATE_LIMIT_MAX_REQUESTS}"
            )
            await _send_json_response(
                send,
                429,
                {
                    "code": ResultCode.IP_BLOCKED.code,
                    "msg": "请求过于频繁，请稍后再试",
                    "data": None,
                },
            )
            return

        await self.app(scope, receive, send)
