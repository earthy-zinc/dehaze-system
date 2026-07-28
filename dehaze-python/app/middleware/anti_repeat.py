"""防重复提交中间件（纯 ASGI 实现）

基于 user_id + method + uri + body_hash 的防重复提交，使用 Redis SET NX EX 实现。
Key 格式：anti_repeat:{user_id}:{method}:{uri}:{body_hash}
仅对 POST/PUT/DELETE 请求生效，TTL 默认 5 秒。
未登录请求退化为基于 client IP，Redis 不可用时降级放行。
"""
import hashlib
import json
import logging

from starlette.types import ASGIApp, Receive, Scope, Send

from app.config import settings
from app.core.code import ResultCode
from app.dependencies.auth import SESSION_COOKIE, SESSION_PREFIX
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

_WRITE_METHODS = {"POST", "PUT", "DELETE"}


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


def _extract_session_id(scope: Scope) -> str | None:
    headers = dict(scope.get("headers", []))
    cookie_header = headers.get(b"cookie", b"").decode("latin-1")
    for cookie in cookie_header.split(";"):
        cookie = cookie.strip()
        if cookie.startswith(f"{SESSION_COOKIE}="):
            return cookie[len(f"{SESSION_COOKIE}="):]
    session_header = headers.get(SESSION_COOKIE.lower().encode("latin-1"))
    if session_header:
        return session_header.decode("latin-1")
    return None


async def _resolve_user_id(scope: Scope) -> str:
    session_id = _extract_session_id(scope)
    if session_id:
        redis = await get_redis_client()
        if redis:
            session_json = await redis_operation_with_fallback(
                operation=lambda: redis.get(f"{SESSION_PREFIX}{session_id}"),
                default=None,
                operation_name="anti_repeat_session_lookup",
            )
            if session_json:
                try:
                    session = json.loads(session_json)
                    user_id = session.get("userId")
                    if user_id is not None:
                        return f"user:{user_id}"
                except (json.JSONDecodeError, TypeError):
                    pass
        return f"session:{session_id}"
    client = scope.get("client")
    return f"ip:{client[0]}" if client else "ip:unknown"


class AntiRepeatMiddleware:
    """防重复提交中间件（SET NX EX）"""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        if not settings.ANTI_REPEAT_ENABLED:
            return await self.app(scope, receive, send)

        method = scope["method"]
        if method not in _WRITE_METHODS:
            return await self.app(scope, receive, send)

        path = scope["path"]
        if path in _EXCLUDE_PATHS:
            return await self.app(scope, receive, send)

        body = await _read_body(receive)
        body_hash = hashlib.md5(body).hexdigest() if body else ""

        user_id = await _resolve_user_id(scope)
        cache_key = f"anti_repeat:{user_id}:{method}:{path}:{body_hash}"

        redis = await get_redis_client()
        if not redis:
            return await self._pass_through(scope, receive, send, body)

        async def _set_nx():
            return await redis.set(cache_key, "1", nx=True, ex=settings.ANTI_REPEAT_TTL_SECONDS)

        ok = await redis_operation_with_fallback(
            operation=_set_nx,
            default=True,
            operation_name=f"anti_repeat:{method}:{path}",
        )

        if not ok:
            logger.warning(
                f"防重复提交拦截: user={user_id}, method={method}, path={path}"
            )
            await _send_json_response(
                send,
                429,
                {
                    "code": ResultCode.REPEAT_SUBMIT_ERROR.code,
                    "msg": ResultCode.REPEAT_SUBMIT_ERROR.msg,
                    "data": None,
                },
            )
            return

        await self._pass_through(scope, receive, send, body)

    async def _pass_through(self, scope: Scope, receive: Receive, send: Send, body: bytes):
        body_sent = False

        async def receive_wrapper():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            return await receive()

        await self.app(scope, receive_wrapper, send)


async def _read_body(receive: Receive) -> bytes:
    body = b""
    more_body = True
    while more_body:
        message = await receive()
        if message["type"] == "http.request":
            body += message.get("body", b"")
            more_body = message.get("more_body", False)
        else:
            break
    return body
