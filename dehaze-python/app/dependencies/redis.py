import asyncio
import logging
import time
from collections.abc import AsyncGenerator

from redis.asyncio import ConnectionPool, Redis

from app.config import settings
from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback

logger = logging.getLogger(__name__)

__all__ = [
    "get_redis",
    "get_redis_client",
    "close_redis",
    "check_redis_health",
]

_redis_pool: ConnectionPool | None = None
_redis_client: Redis | None = None


class RedisHealthStatus:
    def __init__(
        self,
        healthy: bool,
        message: str = "",
        latency_ms: float | None = None,
    ):
        self.healthy = healthy
        self.message = message
        self.latency_ms = latency_ms

    def to_dict(self) -> dict[str, bool | str | float]:
        result: dict[str, bool | str | float] = {
            "healthy": self.healthy,
            "message": self.message,
        }
        if self.latency_ms is not None:
            result["latency_ms"] = round(self.latency_ms, 2)
        return result


def _get_connection_pool_kwargs() -> dict:
    return {
        "max_connections": settings.REDIS_MAX_CONNECTIONS,
        "socket_timeout": settings.REDIS_SOCKET_TIMEOUT,
        "socket_connect_timeout": settings.REDIS_SOCKET_CONNECT_TIMEOUT,
        "retry_on_timeout": settings.REDIS_RETRY_ON_TIMEOUT,
        "health_check_interval": settings.REDIS_HEALTH_CHECK_INTERVAL,
    }


async def _get_redis_pool() -> ConnectionPool:
    global _redis_pool
    if _redis_pool is None:
        kwargs = _get_connection_pool_kwargs()
        logger.info(
            f"创建 Redis 连接池: host={settings.REDIS_HOST}, "
            f"port={settings.REDIS_PORT}, db={settings.REDIS_DB}, "
            f"max_connections={kwargs['max_connections']}"
        )
        _redis_pool = ConnectionPool.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            **kwargs,
        )
    return _redis_pool


async def check_redis_health() -> RedisHealthStatus:
    """
    检查 Redis 连接健康状态并输出日志

    通过 redis_operation_with_fallback 统一处理异常和降级。
    日志输出由本函数内部处理，调用方无需关心细节。
    """

    async def _ping() -> RedisHealthStatus:
        pool = await _get_redis_pool()
        client = Redis(connection_pool=pool, encoding="utf-8", decode_responses=True)
        try:
            start = time.monotonic()
            await asyncio.wait_for(client.ping(), timeout=settings.REDIS_SOCKET_CONNECT_TIMEOUT)
            latency = (time.monotonic() - start) * 1000
            return RedisHealthStatus(
                healthy=True,
                message="Redis connection is healthy",
                latency_ms=latency,
            )
        finally:
            await client.aclose()

    result = await redis_operation_with_fallback(
        operation=_ping,
        default=RedisHealthStatus(healthy=False, message="Redis unavailable"),
        operation_name="health_check",
    )

    if result is not None and result.healthy:
        logger.info(f"Redis 连接正常: 延迟={result.latency_ms:.2f}ms")
    else:
        msg = result.message if result else "Redis unavailable"
        logger.warning(f"Redis 连接异常: {msg}")
        if not settings.DEBUG:
            logger.error("Redis 不可用，部分功能(验证码、缓存等)将无法使用")

    return result or RedisHealthStatus(healthy=False, message="Redis unavailable")


async def get_redis() -> AsyncGenerator[Redis, None]:
    """
    获取 Redis 连接（依赖注入）

    使用生成器确保连接正确关闭

    Yields:
        Redis: 异步 Redis 客户端
    """
    client = Redis(
        connection_pool=await _get_redis_pool(),
        encoding="utf-8",
        decode_responses=True,
    )
    try:
        yield client
    finally:
        await client.aclose()


async def get_redis_client() -> Redis:
    """
    获取全局 Redis 客户端（单例模式）

    用于非依赖注入场景，如后台任务、认证中间件

    Returns:
        Redis: 异步 Redis 客户端
    """
    global _redis_client
    if _redis_client is None:
        pool = await _get_redis_pool()
        _redis_client = Redis(connection_pool=pool, encoding="utf-8", decode_responses=True)
    return _redis_client


async def close_redis():
    global _redis_pool
    if _redis_pool:
        logger.info("关闭 Redis 连接池")
        await _redis_pool.disconnect()
        _redis_pool = None
