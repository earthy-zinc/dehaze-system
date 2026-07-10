import asyncio
import functools
import logging
import time
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union, cast

from redis.exceptions import ConnectionError, RedisError, TimeoutError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RedisUnavailableError(Exception):
    def __init__(self, message: str = "Redis is unavailable"):
        self.message = message
        super().__init__(self.message)


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
            logger.error(f"降级操作失败 [{context}]: {e}")
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
        logger.warning(f"Redis 连接异常 [{operation_name}]: {e}")
        return await _try_fallback(fallback, default, operation_name)
    except RedisError as e:
        logger.error(f"Redis 操作异常 [{operation_name}]: {e}")
        return await _try_fallback(fallback, default, operation_name)


def with_redis_retry(
    max_retries: int = 3,
    retry_delay: float = 0.1,
    fallback_value: Optional[Any] = None,
):
    """
    Redis 操作重试装饰器

    Args:
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）
        fallback_value: 重试失败后的返回值

    Example:
        @with_redis_retry(max_retries=3, fallback_value=None)
        async def get_cache(redis: Redis, key: str):
            return await redis.get(key)
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[BaseException] = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (ConnectionError, TimeoutError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        logger.debug(
                            f"Redis 操作重试 [{func.__name__}]: "
                            f"attempt={attempt + 1}/{max_retries}"
                        )
                    continue
                except RedisError as e:
                    logger.error(f"Redis 操作失败 [{func.__name__}]: {e}")
                    return cast(T, fallback_value)

            logger.error(
                f"Redis 操作最终失败 [{func.__name__}]: "
                f"retries={max_retries}, error={last_exception}"
            )
            return cast(T, fallback_value)

        return cast(Callable[..., Awaitable[T]], wrapper)

    return decorator


class RedisCircuitBreaker:
    """
    Redis 熔断器

    当 Redis 连续失败达到阈值时，熔断器打开，后续请求直接返回默认值。
    经过冷却时间后，熔断器进入半开状态，尝试恢复连接。

    状态流转:
    CLOSED -> (失败次数达到阈值) -> OPEN -> (冷却时间后) -> HALF_OPEN
    HALF_OPEN -> (成功) -> CLOSED
    HALF_OPEN -> (失败) -> OPEN
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        name: str = "redis",
    ):
        """
        初始化熔断器

        Args:
            failure_threshold: 失败阈值，连续失败次数达到此值时熔断
            recovery_timeout: 恢复超时时间（秒），熔断后等待此时间尝试恢复
            name: 熔断器名称，用于日志
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name

        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = "CLOSED"

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_open(self) -> bool:
        if self._state == "OPEN":
            if (
                self._last_failure_time
                and time.monotonic() - self._last_failure_time >= self.recovery_timeout
            ):
                self._state = "HALF_OPEN"
                logger.info(f"[{self.name}] 熔断器进入半开状态")
                return False
            return True
        return False

    def record_success(self):
        self._failure_count = 0
        if self._state != "CLOSED":
            self._state = "CLOSED"
            logger.info(f"[{self.name}] 熔断器恢复关闭状态")

    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == "HALF_OPEN":
            self._state = "OPEN"
            logger.warning(f"[{self.name}] 熔断器重新打开")
        elif self._failure_count >= self.failure_threshold:
            self._state = "OPEN"
            logger.warning(
                f"[{self.name}] 熔断器打开: "
                f"failures={self._failure_count}, threshold={self.failure_threshold}"
            )

    async def call(
        self,
        operation: Callable[[], Awaitable[T]],
        fallback: Optional[Callable[[], Awaitable[T]]] = None,
        default: Optional[T] = None,
    ) -> Optional[T]:
        """
        通过熔断器执行操作

        Args:
            operation: 要执行的操作
            fallback: 降级操作
            default: 默认返回值

        Returns:
            操作结果或默认值
        """
        if self.is_open:
            return await _try_fallback(fallback, default, self.name)

        async def wrapped_operation() -> T:
            result = await operation()
            self.record_success()
            return result

        try:
            return await wrapped_operation()
        except (ConnectionError, TimeoutError) as e:
            self.record_failure()
            logger.warning(f"[{self.name}] Redis 连接异常: {e}")
            return await _try_fallback(fallback, default, self.name)
        except RedisError as e:
            logger.error(f"[{self.name}] Redis 操作异常: {e}")
            return await _try_fallback(fallback, default, self.name)


_global_circuit_breaker: Optional[RedisCircuitBreaker] = None


def get_redis_circuit_breaker() -> RedisCircuitBreaker:
    global _global_circuit_breaker
    if _global_circuit_breaker is None:
        _global_circuit_breaker = RedisCircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            name="redis_global",
        )
    return _global_circuit_breaker
