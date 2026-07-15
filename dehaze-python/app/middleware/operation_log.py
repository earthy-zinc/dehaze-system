"""
操作日志中间件（纯 ASGI 实现）

相比 BaseHTTPMiddleware 的优势：
- 不会将整个请求体读入内存，通过 tee receive 流式采集
- 正确处理 StreamingResponse（不缓冲完整响应体）
- 不创建 Request/Response 包装对象，减少开销
- 与 ASGI 生命周期兼容，不会出现 edge case
"""

import asyncio
import json
import logging
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.database import get_db_session
from app.models.base import get_current_user_id
from app.models.entity.sys_log import SysOperationLog

_logger = logging.getLogger(__name__)

EXCLUDE_PATHS = {
    "/health",
    "/health/db",
    "/health/redis",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
    "/metrics",
}

SENSITIVE_FIELDS = {
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "cookie",
}

# 响应体采集上限（字节），超出部分截断
_MAX_RESPONSE_CAPTURE = 500

# 敏感数据递归过滤最大深度
_MAX_FILTER_DEPTH = 5


def filter_sensitive_data(data: dict | str | None) -> str:
    """过滤敏感数据（递归过滤嵌套字典中的敏感字段）"""
    if not data:
        return ""

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return data[:500]

    if not isinstance(data, dict):
        return str(data)[:500]

    return json.dumps(_filter_dict(data, depth=0), ensure_ascii=False)


def _filter_dict(data: dict, depth: int) -> dict:
    """递归过滤字典中的敏感字段"""
    if depth >= _MAX_FILTER_DEPTH:
        return {k: "..." for k in data}

    filtered = {}
    for key, value in data.items():
        if key.lower() in SENSITIVE_FIELDS:
            filtered[key] = "******"
        elif isinstance(value, dict):
            filtered[key] = _filter_dict(value, depth + 1)
        elif isinstance(value, str) and len(value) > 200:
            filtered[key] = value[:200] + "..."
        else:
            filtered[key] = value
    return filtered


def _is_streaming_response(headers: list[tuple[bytes, bytes]]) -> bool:
    """根据响应头判断是否为流式响应"""
    for name, value in headers:
        try:
            decoded_name = name.decode("latin-1").lower()
            decoded_value = value.decode("latin-1").lower()
            if decoded_name == "content-type" and (
                "text/event-stream" in decoded_value
                or "octet-stream" in decoded_value
            ):
                return True
            if decoded_name == "transfer-encoding" and "chunked" in decoded_value:
                return True
        except (UnicodeDecodeError, AttributeError) as e:
            _logger.debug(f"解析响应头失败: {e}")
    return False


class OperationLogMiddleware:
    """纯 ASGI 操作日志中间件"""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        # 仅处理 HTTP 请求，WebSocket / lifespan 直接透传
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path: str = scope["path"]
        if path in EXCLUDE_PATHS or path.startswith("/static"):
            return await self.app(scope, receive, send)

        method: str = scope["method"]
        start_time = time.time()

        # ===== Tee receive：流式采集请求体 =====
        body_chunks: list[bytes] = []

        async def receive_tee() -> Message:
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                if body:
                    body_chunks.append(body)
            return message

        # ===== Tee send：采集响应状态码和响应体 =====
        response_status = 0
        response_headers: list[tuple[bytes, bytes]] = []
        response_body_chunks: list[bytes] = []
        is_streaming = False
        captured_size = 0

        async def send_tee(message: Message):
            nonlocal response_status, response_headers, is_streaming, captured_size

            if message["type"] == "http.response.start":
                response_status = message["status"]
                response_headers = message.get("headers", [])
                is_streaming = _is_streaming_response(response_headers)
            elif message["type"] == "http.response.body":
                if not is_streaming and captured_size < _MAX_RESPONSE_CAPTURE:
                    body = message.get("body", b"")
                    if body:
                        remaining = _MAX_RESPONSE_CAPTURE - captured_size
                        response_body_chunks.append(body[:remaining])
                        captured_size += min(len(body), remaining)

            await send(message)

        # ===== 执行请求 =====
        try:
            await self.app(scope, receive_tee, send_tee)
        except Exception:
            response_status = 500
            raise
        finally:
            latency = int((time.time() - start_time) * 1000)

            # 解析请求体
            request_body = ""
            if method in ("POST", "PUT", "PATCH") and body_chunks:
                raw_body = b"".join(body_chunks)
                try:
                    body_data = json.loads(raw_body)
                    request_body = filter_sensitive_data(body_data)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    request_body = raw_body.decode("utf-8", errors="ignore")[:500]

            # 解析响应体
            response_body = ""
            if is_streaming:
                response_body = "[Streaming Response]"
            elif response_body_chunks:
                try:
                    raw = b"".join(response_body_chunks)
                    response_body = raw.decode("utf-8", errors="ignore")
                except Exception:
                    pass

            # 从 scope 提取客户端信息
            client = scope.get("client")
            ip = client[0] if client else ""

            user_agent = ""
            for name, value in scope.get("headers", []):
                if name == b"user-agent":
                    user_agent = value.decode("latin-1", errors="ignore")
                    break

            # 从 contextvar 获取 user_id（由 auth 依赖注入设置）
            user_id = get_current_user_id()

            # 后台异步保存日志（不阻塞响应）
            task = asyncio.create_task(
                _save_log_safe(
                    ip=ip,
                    method=method,
                    path=path,
                    status=response_status,
                    latency=latency,
                    agent=user_agent,
                    body=request_body,
                    resp=response_body,
                    user_id=user_id,
                )
            )
            task.add_done_callback(_handle_task_exception)


def _handle_task_exception(task: asyncio.Task):
    try:
        task.result()
    except Exception as e:
        _logger.warning(f"后台日志保存任务失败: {e}")


async def _save_log_safe(**kwargs):
    try:
        await _save_log(**kwargs)
    except Exception as e:
        _logger.warning(f"保存操作日志失败: {e}")


async def _save_log(
    ip: str,
    method: str,
    path: str,
    status: int,
    latency: int,
    agent: str,
    body: str,
    resp: str,
    user_id: int | None,
):
    async with get_db_session() as session:
        log = SysOperationLog(
            ip=ip,
            method=method,
            path=path,
            status=status,
            latency=latency,
            agent=agent[:512] if agent else "",
            body=body[:500] if body else "",
            resp=resp[:500] if resp else "",
            user_id=user_id,
        )
        session.add(log)
