"""入站鉴权：服务间调用统一 Authorization: Bearer ${INTERNAL_SERVICE_KEY}。

用纯 ASGI 中间件而非 BaseHTTPMiddleware：后者在响应外层再包一层任务组，对流式响应的
断连取消语义有干扰，而流式对话必须保持"客户端断连即停止生成"。
密钥是部署级静态凭证（env 注入，代表服务身份），与用户 API Key 无关；
比较用 hmac.compare_digest 防时序侧信道；密钥未配置时 fail closed。
"""

import hmac

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from app.config import settings
from app.errors import unauthorized_response


class BearerAuthMiddleware:
    """全部端点（含 /health、/ready、/metrics）的入站鉴权，无免鉴权例外"""

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
