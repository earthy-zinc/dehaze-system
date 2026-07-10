import asyncio
import json
import logging
import time
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.database import get_db_session
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


def filter_sensitive_data(data: dict | str | None) -> str:
    """
    过滤敏感数据

    Args:
        data: 原始数据

    Returns:
        过滤后的 JSON 字符串
    """
    if not data:
        return ""

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return data[:500]  # 截断非 JSON 数据

    if not isinstance(data, dict):
        return str(data)[:500]

    filtered = {}
    for key, value in data.items():
        if key.lower() in SENSITIVE_FIELDS:
            filtered[key] = "******"
        elif isinstance(value, dict):
            filtered[key] = json.dumps(
                {k: "******" if k.lower() in SENSITIVE_FIELDS else v for k, v in value.items()},
                ensure_ascii=False,
            )
        elif isinstance(value, str) and len(value) > 200:
            filtered[key] = value[:200] + "..."
        else:
            filtered[key] = value

    return json.dumps(filtered, ensure_ascii=False)


class OperationLogMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in EXCLUDE_PATHS or request.url.path.startswith("/static"):
            return await call_next(request)

        start_time = time.time()

        # request.body() 只能读取一次，需提前提取
        request_body = ""
        if request.method in ("POST", "PUT", "PATCH"):
            try:
                body_bytes = await request.body()
                if body_bytes:
                    try:
                        body_data = json.loads(body_bytes)
                        request_body = filter_sensitive_data(body_data)
                    except json.JSONDecodeError:
                        request_body = body_bytes.decode("utf-8", errors="ignore")[:500]
            except Exception:
                pass

        response = await call_next(request)

        latency = int((time.time() - start_time) * 1000)

        # 区分普通响应和流式响应
        response_body = ""
        if isinstance(response, (StreamingResponse, FileResponse)):
            response_body = "[Streaming Response]"
        elif hasattr(response, "body"):
            try:
                body = response.body
                if isinstance(body, bytes):
                    body = body.decode("utf-8", errors="ignore")
                response_body = body[:500]
            except Exception:
                pass

        user_id = getattr(request.state, "user_id", None)

        task = asyncio.create_task(
            self._save_log_safe(
                ip=request.client.host if request.client else "",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                latency=latency,
                agent=request.headers.get("user-agent", ""),
                body=request_body,
                resp=response_body,
                user_id=user_id,
            )
        )
        task.add_done_callback(self._handle_task_exception)

        return response

    def _handle_task_exception(self, task: asyncio.Task):
        try:
            task.result()
        except Exception as e:
            _logger.warning(f"后台日志保存任务失败: {e}")

    async def _save_log_safe(self, **kwargs):
        try:
            await self._save_log(**kwargs)
        except Exception as e:
            _logger.warning(f"保存操作日志失败: {e}")

    async def _save_log(
        self,
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
