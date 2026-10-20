"""入站鉴权：服务间调用统一 Authorization: Bearer ${INTERNAL_SERVICE_KEY}。

用纯 ASGI 中间件而非 Route 级依赖：鉴权必须在读取请求体/进入路由前短路，本服务有 multipart
音频上传与 WS 上行，ASGI 层能在不改动流式与断连语义的前提下拒绝（对齐 clarity-llm 形态）。
密钥是部署级静态凭证（env 注入，代表服务身份），与用户 API Key 无关；WebSocket 同样以握手头
携带（不走 query，避免凭证进访问日志），由路由在握手阶段判定。
比较用 hmac.compare_digest 防时序侧信道；密钥未配置时 fail closed。
"""

from __future__ import annotations

import hmac

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.config import settings

_BEARER_PREFIX = "Bearer "


def unauthorized_response() -> JSONResponse:
    """统一的 401 响应体"""
    return JSONResponse(
        status_code=401,
        content={"code": "UNAUTHORIZED", "message": "未授权：内部服务密钥缺失或错误"},
    )


def _authorized(authorization: str | None) -> bool:
    key = settings.INTERNAL_SERVICE_KEY
    # 密钥未配置时 fail closed：不因空密钥退化为无鉴权
    if not key or not authorization or not authorization.startswith(_BEARER_PREFIX):
        return False
    # 常量时间比较，防逐位猜测
    token = authorization[len(_BEARER_PREFIX) :]
    return hmac.compare_digest(token.encode("utf-8"), key.encode("utf-8"))


def authorize_websocket(authorization: str | None) -> bool:
    """WS 握手鉴权：失败由调用方拒绝握手（不向未认证连接暴露协议细节）"""
    return _authorized(authorization)


class BearerAuthMiddleware:
    """全部 HTTP 端点的入站鉴权（含 /health、/ready、/metrics，无免鉴权例外）"""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and not _authorized(Headers(scope=scope).get("authorization")):
            await unauthorized_response()(scope, receive, send)
            return
        await self.app(scope, receive, send)
