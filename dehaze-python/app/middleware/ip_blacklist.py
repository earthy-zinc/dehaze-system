"""
IP 黑名单中间件（纯 ASGI 实现）

功能：
1. 检查请求 IP 是否在 Redis 黑名单中，命中则直接返回 403
2. 自动追踪异常请求（4xx/5xx），超过阈值时自动封禁 IP
3. 支持手动管理黑名单（通过 IPBlacklistService）

Redis key 设计：
- ip:blacklist:{ip} → 存在即封禁，带 TTL
- ip:errors:{ip}    → sorted set，score=时间戳，追踪窗口内的异常请求
"""

import asyncio
import json
import logging
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import settings
from app.core.code import ResultCode
from app.dependencies.redis import get_redis_client
from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback

logger = logging.getLogger(__name__)

# 不参与追踪的路径
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

# 内网 IP 前缀（开发环境跳过）
_INTERNAL_IP_PREFIXES = ("127.", "10.", "192.168.", "172.16.", "172.17.", "172.18.",
                         "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
                         "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
                         "172.29.", "172.30.", "172.31.", "::1", "localhost")


def _is_internal_ip(ip: str) -> bool:
    return any(ip.startswith(prefix) for prefix in _INTERNAL_IP_PREFIXES)


async def _send_json_response(send: Send, status_code: int, content: dict):
    """直接通过 ASGI send 发送 JSON 响应（不经过应用层）"""
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


class IPBlacklistMiddleware:
    """IP 黑名单中间件（纯 ASGI）"""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        if not settings.IP_BLACKLIST_ENABLED:
            return await self.app(scope, receive, send)

        path: str = scope["path"]
        if path in _EXCLUDE_PATHS:
            return await self.app(scope, receive, send)

        # 获取客户端 IP
        client = scope.get("client")
        if not client:
            return await self.app(scope, receive, send)

        ip = client[0]

        # 开发环境跳过内网 IP
        if settings.DEBUG and _is_internal_ip(ip):
            return await self.app(scope, receive, send)

        # 检查黑名单
        redis = await get_redis_client()
        if redis:
            blacklisted = await redis_operation_with_fallback(
                operation=lambda: redis.exists(f"ip:blacklist:{ip}"),
                default=0,
                operation_name="ip_blacklist_check",
            )
            if blacklisted:
                logger.warning(f"IP 被封禁拦截: ip={ip}, path={path}")
                await _send_json_response(
                    send,
                    403,
                    {
                        "code": ResultCode.IP_BLOCKED.code,
                        "msg": ResultCode.IP_BLOCKED.msg,
                        "data": None,
                    },
                )
                return

        # 追踪响应状态码
        response_status = 0

        async def send_wrapper(message: Message):
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # 异常请求追踪（4xx/5xx）
            if response_status >= 400 and redis:
                asyncio.create_task(_track_ip_error(ip, response_status, redis))


async def _track_ip_error(ip: str, status: int, redis):
    """追踪 IP 的异常请求，超过阈值自动封禁"""
    try:
        now = time.time()
        key = f"ip:errors:{ip}"

        pipe = redis.pipeline()
        pipe.zadd(key, {f"{now}:{status}": now})
        pipe.zremrangebyscore(key, 0, now - settings.IP_BLACKLIST_TRACKING_WINDOW)
        pipe.zcard(key)
        pipe.expire(key, settings.IP_BLACKLIST_TRACKING_WINDOW)
        results = await pipe.execute()

        error_count = results[2]

        if error_count >= settings.IP_BLACKLIST_THRESHOLD:
            await redis.setex(
                f"ip:blacklist:{ip}",
                settings.IP_BLACKLIST_DURATION,
                str(int(now)),
            )
            # 清空错误记录
            await redis.delete(key)
            logger.warning(
                f"IP 自动封禁: ip={ip}, errors={error_count}, "
                f"duration={settings.IP_BLACKLIST_DURATION}s"
            )
    except Exception as e:
        logger.warning(f"IP 错误追踪失败: ip={ip}, error={e}")


class IPBlacklistService:
    """IP 黑名单管理服务（供管理端 API 调用）"""

    @staticmethod
    async def add(ip: str, duration: int | None = None) -> bool:
        """手动添加 IP 到黑名单"""
        redis = await get_redis_client()
        if not redis:
            return False
        ttl = duration or settings.IP_BLACKLIST_DURATION
        await redis.setex(f"ip:blacklist:{ip}", ttl, str(int(time.time())))
        logger.info(f"IP 手动封禁: ip={ip}, duration={ttl}s")
        return True

    @staticmethod
    async def remove(ip: str) -> bool:
        """从黑名单移除 IP"""
        redis = await get_redis_client()
        if not redis:
            return False
        await redis.delete(f"ip:blacklist:{ip}")
        await redis.delete(f"ip:errors:{ip}")
        logger.info(f"IP 解封: ip={ip}")
        return True

    @staticmethod
    async def is_blocked(ip: str) -> bool:
        """检查 IP 是否被封禁"""
        redis = await get_redis_client()
        if not redis:
            return False
        return bool(await redis.exists(f"ip:blacklist:{ip}"))

    @staticmethod
    async def get_error_count(ip: str) -> int:
        """获取 IP 当前窗口内的异常请求次数"""
        redis = await get_redis_client()
        if not redis:
            return 0
        now = time.time()
        key = f"ip:errors:{ip}"
        await redis.zremrangebyscore(key, 0, now - settings.IP_BLACKLIST_TRACKING_WINDOW)
        return await redis.zcard(key)
