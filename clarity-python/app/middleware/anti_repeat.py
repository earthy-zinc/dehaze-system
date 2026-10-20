"""防重复提交中间件（纯 ASGI 实现）

键格式（三端共享同一 Redis，**格式即契约**，设计见
`clarity-doc/docs/05-改造计划/防重复提交三端统一改造计划.md`）：

    anti_repeat:<identity>:<METHOD>:<path>:<sorted query>:<bodyHash>

- `identity`：`user:{userId}`（会话可取时）；退化 `session:{sid}` → `ip:{clientIp}`。
  身份必须稳定跨登录态，否则"重新登录即绕过防重"。
- `sorted_query`：query 参数按 `(key, value)` 全排序后重新编码，保证 `?a=1&b=2` 与
  `?b=2&a=1` 命中同一键（不排序等于留一条绕过路径）；无参数记 `""`。
- `bodyHash`：`md5(rawBody).hexdigest()`，空 body 记 `""`。

仅对 **POST** 请求生效（勿按 docstring 旧述的 POST/PUT/DELETE 理解），TTL 默认 5 秒。
Redis 不可用时降级放行。
"""

import hashlib
import json
import logging
import re
from urllib.parse import parse_qsl, urlencode

from starlette.types import ASGIApp, Receive, Scope, Send

from app.config import settings
from app.core.code import ResultCode
from app.dependencies.auth import SESSION_COOKIE, SESSION_PREFIX
from app.dependencies.redis import get_redis_client
from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback
from app.middleware._shared import send_json_response

logger = logging.getLogger(__name__)

# 防重端点白名单（三端必须**完全一致**，由 `tests/test_anti_repeat_contract.py`
# 对拍 java 注解与 go 路由挂载）。
# 入选标准：创建/导入类，且"同一窗口内重复提交同一内容"必为误操作。
# 不入选：
#   - 只读查询（POST `/{module}/_export` 仅因筛选条件复杂而用 POST）；
#   - multipart 上传（请求体无法有效摘要）且"连续操作"合法（`/item-files` 连续传多图、
#     `/dataset-items*` 连续建项、`/files` 连续上传、`/image-input/history` 高频写入）。
# 白名单形式（而非"全局拦截 + 排除前缀"）是三端集合可严格相等、可守卫的前提，
# 也避免"排除整个前缀"连带放过该前缀下其他 POST。
_PROTECTED_PATHS = {
    "/api/v1/depts",
    "/api/v1/dict",
    "/api/v1/dict/types",
    "/api/v1/menus",
    "/api/v1/roles",
    "/api/v1/users",
    "/api/v1/announcements",
}

# 参数化白名单：导入端点。模块闭集与 go `supportedImportExportModules`、
# 本仓 `import_export.py` 的 `_ensure_module_supported` 三处一致。
_IMPORT_MODULES = ("user", "role", "dept", "menu", "dict", "dataset")
_PROTECTED_PATTERNS = (re.compile(rf"^/api/v1/(?:{'|'.join(_IMPORT_MODULES)})/_import$"),)


def normalize_query(query_string: str) -> str:
    """query 参数按 (key, value) 全排序后重新编码。

    不排序等于留一条绕过路径：`?a=1&b=2` 与 `?b=2&a=1` 会算出不同键。
    空串 → `""`（与 java/go 口径一致）。多值参数按值排序，保证三端同串。
    """
    if not query_string:
        return ""
    return urlencode(sorted(parse_qsl(query_string, keep_blank_values=True)))


def is_protected(path: str) -> bool:
    """是否属于防重白名单（仅 POST 会走到这里）。"""
    return path in _PROTECTED_PATHS or any(pattern.match(path) for pattern in _PROTECTED_PATTERNS)


def _extract_session_id(scope: Scope) -> str | None:
    headers = dict(scope.get("headers", []))
    cookie_header = headers.get(b"cookie", b"").decode("latin-1")
    for cookie in cookie_header.split(";"):
        cookie = cookie.strip()
        if cookie.startswith(f"{SESSION_COOKIE}="):
            return cookie[len(f"{SESSION_COOKIE}=") :]
    session_header = headers.get(SESSION_COOKIE.lower().encode("latin-1"))
    if session_header:
        return session_header.decode("latin-1")
    return None


async def _resolve_identity(scope: Scope) -> str:
    """身份口径：`user:{userId}`，取不到则退化为 `ip:{clientIp}`。

    不按 session 分键：java 曾用 sessionId、go 曾用 JWT jti，二者都随登录态变化
    ⇒ 在同一用户重新登录后提交同一请求会各自算出新键而绕过防重。统一收敛到用户维度。
    """
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
                    user_id = json.loads(session_json).get("userId")
                    if user_id is not None:
                        return f"user:{user_id}"
                except (json.JSONDecodeError, TypeError, AttributeError):
                    # 会话缓存损坏：退化为按 IP 限流（粒度变粗但仍在防重），需可见以暴露数据异常
                    logger.warning(
                        "防重复提交会话数据解析失败，退化为按 client IP 限流: session=%s",
                        session_id,
                        exc_info=True,
                    )
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
        if method != "POST":
            return await self.app(scope, receive, send)

        path = scope["path"]
        # 白名单外直接透传原始 receive：既不读 body（零开销），也不重建请求流
        if not is_protected(path):
            return await self.app(scope, receive, send)

        body = await _read_body(receive)
        body_hash = hashlib.md5(body).hexdigest() if body else ""

        query = normalize_query(scope.get("query_string", b"").decode("latin-1"))

        identity = await _resolve_identity(scope)
        cache_key = f"anti_repeat:{identity}:{method}:{path}:{query}:{body_hash}"

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
            logger.warning(f"防重复提交拦截: identity={identity}, method={method}, path={path}")
            await send_json_response(
                send,
                429,
                {
                    "code": ResultCode.REPEAT_SUBMIT_ERROR.code,
                    "msg": ResultCode.REPEAT_SUBMIT_ERROR.msg,
                    "data": None,
                },
            )
            return None

        await self._pass_through(scope, receive, send, body)
        return None

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
