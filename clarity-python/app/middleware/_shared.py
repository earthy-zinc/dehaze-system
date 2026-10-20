"""中间件共享工具（收敛跨文件的重复常量与辅助函数）"""

import json

from starlette.types import Send

# 健康检查 / 指标 / 文档等噪声路径，不参与拦截/追踪
EXCLUDE_PATHS = frozenset(
    {
        "/health",
        "/health/db",
        "/health/redis",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico",
    }
)


async def send_json_response(send: Send, status_code: int, content: dict):
    """直接通过 ASGI send 发送 JSON 响应（不经过应用层）"""
    body = json.dumps(content, ensure_ascii=False).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode("latin-1")),
            ],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": body,
        }
    )
