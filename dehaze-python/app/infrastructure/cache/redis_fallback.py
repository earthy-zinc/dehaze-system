"""Redis 操作降级工具

提供：
- RedisCircuitBreaker: 熔断器（CLOSED → OPEN → HALF_OPEN），连续失败达阈值后熔断
- redis_operation_with_fallback: 优雅降级入口（集成熔断器 + 短暂重试 + fallback）
- Prometheus 指标：熔断器状态、降级触发次数、重试次数
"""

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import TypeVar

from prometheus_client import Counter, Gauge
from redis.exceptions import ConnectionError, RedisError, TimeoutError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 连接级异常（ConnectionError/TimeoutError）的自动重试次数
_MAX_RETRIES = 1
_RETRY_BACKOFF = 0.1  # 秒

# ── Prometheus 指标 ──────────────────────────────────────────

REDIS_CIRCUIT_STATE = Gauge(
    "dehaze_redis_circuit_breaker_state",
    "Redis circuit breaker state (0=CLOSED, 1=OPEN, 2=HALF_OPEN)",
    multiprocess_mode="all",  # 各 Worker 独立熔断器，聚合时保留所有进程状态
)

REDIS_FALLBACK_TOTAL = Counter(
    "dehaze_redis_fallback_total",
    "Total number of Redis fallback/degradation triggers",
    ["operation", "reason"],  # reason: connection_error / timeout / redis_error / circuit_open
)

REDIS_RETRY_TOTAL = Counter(
    "dehaze_redis_retry_total",
    "Total number of Redis operation retries",
    ["operation"],
)


# ── 熔断器 ──────────────────────────────────────────────────


class CircuitState(Enum):
    CLOSED = 0  # 正常通行
    OPEN = 1  # 熔断，快速失败
    HALF_OPEN = 2  # 试探恢复


class RedisCircuitBreaker:
    """Redis 熔断器

    状态转换：
    - CLOSED: 正常状态，连续失败计数达 failure_threshold → OPEN
    - OPEN: 熔断状态，所有请求快速失败；经过 recovery_timeout 后 → HALF_OPEN
    - HALF_OPEN: 试探状态，允许单次请求通过；成功 → CLOSED，失败 → OPEN
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ):
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls
        self._half_open_calls = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def allow_request(self) -> bool:
        """判断当前是否允许请求通过"""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                    self._transition_to(CircuitState.HALF_OPEN)
                    self._half_open_calls = 0
                    return True
                return False
            # HALF_OPEN: 限制试探次数
            if self._half_open_calls < self._half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

    async def record_success(self) -> None:
        """记录成功"""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0

    async def record_failure(self) -> None:
        """记录失败"""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if (
                self._state == CircuitState.HALF_OPEN
                or self._failure_count >= self._failure_threshold
            ):
                self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """状态转换并更新 Prometheus 指标"""
        old_state = self._state
        self._state = new_state
        REDIS_CIRCUIT_STATE.set(new_state.value)
        if old_state != new_state:
            logger.info(
                "Redis 熔断器状态变更: %s → %s",
                old_state.name,
                new_state.name,
            )
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0


# 全局熔断器实例
_circuit_breaker = RedisCircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30.0,
)


# ── 降级入口 ──────────────────────────────────────────────


async def _try_fallback(
    fallback: Callable[[], Awaitable[T]] | None,
    default: T | None,
    context: str,
) -> T | None:
    """尝试执行降级操作，失败则返回默认值"""
    if fallback:
        try:
            return await fallback()
        except Exception as e:
            logger.error("降级操作失败 [%s]: %s", context, e)
    return default


def _validate_operation(operation: object, operation_name: str) -> None:
    """入口校验首参是「可无参调用」的可调用对象，不合法时抛出明确的 TypeError。

    传错签名（如带必填参数的函数）时 `operation()` 抛出的 TypeError 不属于 Redis 异常族、
    不会被降级分支捕获，会一路穿透到接口层变成 500，把编码错误伪装成"服务异常"。
    此处前置拒绝并指出调用约束，错误在开发期即暴露。
    """
    if not callable(operation):
        raise TypeError(
            f"redis_operation_with_fallback 的 operation 必须是可调用对象"
            f"（operation_name={operation_name}）：实际为 {type(operation).__name__}"
        )
    try:
        signature = inspect.signature(operation)
    except (TypeError, ValueError):
        # 无法内省签名的可调用对象（部分内置/扩展实现）：跳过零参校验，
        # 由调用处的 awaitable 校验兜底
        return
    # bind() 按"零参调用"试绑定：带必填参数（含带参 partial）在此失败，且不执行函数
    try:
        signature.bind()
    except TypeError as e:
        raise TypeError(
            f"redis_operation_with_fallback 的 operation 必须无参调用"
            f"（operation_name={operation_name}）：实际签名为 {signature}"
        ) from e


async def redis_operation_with_fallback(
    operation: Callable[[], Awaitable[T]],
    fallback: Callable[[], Awaitable[T]] | None = None,
    operation_name: str = "redis_operation",
    default: T | None = None,
) -> T | None:
    """
    执行 Redis 操作，集成熔断器 + 短暂重试 + 优雅降级

    流程：
    1. 检查熔断器状态，OPEN 时快速失败走降级
    2. 执行操作；ConnectionError/TimeoutError 自动重试 1 次（0.1s 退避）
    3. 重试仍失败或其他 RedisError → 记录熔断器失败 + 走降级

    Args:
        operation: Redis 操作（**无参**可调用对象：async 函数，或返回 awaitable 的
            零参可调用对象如 `lambda: redis.get(key)`）
        fallback: 降级操作函数（异步），Redis 不可用时执行
        operation_name: 操作名称，用于日志和指标
        default: 默认返回值，当没有 fallback 时使用

    Returns:
        操作结果，或降级结果/默认值
    """
    _validate_operation(operation, operation_name)

    # 熔断器检查：OPEN 状态快速失败
    if not await _circuit_breaker.allow_request():
        REDIS_FALLBACK_TOTAL.labels(operation=operation_name, reason="circuit_open").inc()
        logger.warning("Redis 熔断器已开启，快速降级 [%s]", operation_name)
        return await _try_fallback(fallback, default, operation_name)

    # 带重试的执行（仅对连接级异常重试）
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            returned = operation()
            if not inspect.isawaitable(returned):
                # 同步函数/返回非 awaitable：同样不属于 Redis 异常族，显式报错而非静默降级
                raise TypeError(
                    f"redis_operation_with_fallback 的 operation 必须返回 awaitable"
                    f"（operation_name={operation_name}）：实际返回 {type(returned).__name__}"
                )
            result = await returned
            await _circuit_breaker.record_success()
            return result
        except (ConnectionError, TimeoutError) as e:
            last_exc = e
            if attempt < _MAX_RETRIES:
                REDIS_RETRY_TOTAL.labels(operation=operation_name).inc()
                logger.warning(
                    "Redis 连接异常，重试 [%s] attempt=%d/%d: %s",
                    operation_name,
                    attempt + 1,
                    _MAX_RETRIES,
                    e,
                )
                await asyncio.sleep(_RETRY_BACKOFF)
        except RedisError as e:
            # 非连接级 RedisError 不重试，直接降级
            await _circuit_breaker.record_failure()
            REDIS_FALLBACK_TOTAL.labels(operation=operation_name, reason="redis_error").inc()
            logger.error("Redis 操作异常 [%s]: %s", operation_name, e)
            return await _try_fallback(fallback, default, operation_name)

    # 重试耗尽，记录失败并降级
    await _circuit_breaker.record_failure()
    reason = "connection_error" if isinstance(last_exc, ConnectionError) else "timeout"
    REDIS_FALLBACK_TOTAL.labels(operation=operation_name, reason=reason).inc()
    logger.warning("Redis 连接异常（重试耗尽）[%s]: %s", operation_name, last_exc)
    return await _try_fallback(fallback, default, operation_name)
