"""入站鉴权：服务间调用统一 Authorization: Bearer ${INTERNAL_SERVICE_KEY}。

用纯 ASGI 中间件而非 FastAPI 依赖：鉴权发生在应用栈最外层，结果图大响应与断连取消语义
不受 BaseHTTPMiddleware 任务组封装干扰；401 响应体在此统一构造，任何路径（探针、/metrics）
口径一致。密钥是部署级静态凭证（代表服务身份），与用户态凭证无关。
"""

import hmac

from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from app.config import settings


def unauthorized_response() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"code": "UNAUTHORIZED", "message": "未授权：内部服务密钥缺失或错误"},
    )


class BearerAuthMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and not _authorized(Headers(scope=scope)):
            await unauthorized_response()(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _authorized(headers: Headers) -> bool:
    key = settings.INTERNAL_SERVICE_KEY
    # 密钥未配置时 fail closed：不因空密钥退化为无鉴权
    if not key:
        return False
    scheme, _, token = headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer":
        return False
    # 常量时间比较，防逐位猜测
    return hmac.compare_digest(token.encode("utf-8"), key.encode("utf-8"))
