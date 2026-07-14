"""Redis 操作降级工具

提供 redis_operation_with_fallback 优雅降级机制：
- Redis 连接异常/超时/操作异常时，自动执行 fallback 或返回 default
"""
import logging
from typing import Awaitable, Callable, Optional, TypeVar

from redis.exceptions import ConnectionError, RedisError, TimeoutError

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def _try_fallback(
    fallback: Optional[Callable[[], Awaitable[T]]],
    default: Optional[T],
    context: str,
) -> Optional[T]:
    """
    尝试执行降级操作，失败则返回默认值

    Args:
        fallback: 降级操作函数（异步）
        default: 默认返回值
        context: 上下文标识，用于日志
    """
    if fallback:
        try:
            return await fallback()
        except Exception as e:
            logger.error("降级操作失败 [%s]: %s", context, e)
    return default


async def redis_operation_with_fallback(
    operation: Callable[[], Awaitable[T]],
    fallback: Optional[Callable[[], Awaitable[T]]] = None,
    operation_name: str = "redis_operation",
    default: Optional[T] = None,
) -> Optional[T]:
    """
    执行 Redis 操作，支持优雅降级

    Args:
        operation: Redis 操作函数（异步）
        fallback: 降级操作函数（异步），Redis 不可用时执行
        operation_name: 操作名称，用于日志
        default: 默认返回值，当没有 fallback 时使用

    Returns:
        操作结果，或降级结果/默认值

    Example:
        value = await redis_operation_with_fallback(
            operation=lambda: redis.get("key"),
            fallback=lambda: get_from_db("key"),
            operation_name="get_cache",
            default=None,
        )
    """
    try:
        return await operation()
    except (ConnectionError, TimeoutError) as e:
        logger.warning("Redis 连接异常 [%s]: %s", operation_name, e)
        return await _try_fallback(fallback, default, operation_name)
    except RedisError as e:
        logger.error("Redis 操作异常 [%s]: %s", operation_name, e)
        return await _try_fallback(fallback, default, operation_name)
