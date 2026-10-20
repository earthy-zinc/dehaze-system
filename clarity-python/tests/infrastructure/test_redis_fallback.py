"""降级入口 `redis_operation_with_fallback` 的首参契约校验。

首参必须是「可无参调用且返回 awaitable」：传带必填参数的函数时 `operation()` 抛出的
TypeError 不属于 Redis 异常族、不会被降级分支捕获，会穿透成接口 500。本用例守住
"在开发期显式失败"，同时守住"零参可调用对象返回 awaitable"（`lambda: redis.get(key)`
写法，cache.py 大量使用）不被误拒。
"""

import pytest

from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback


async def test_rejects_operation_with_required_argument():
    """带必填参数的函数必须显式失败，且错误信息指出"无参"约束（不许穿透）"""

    async def _op(key: str) -> str:
        return key

    with pytest.raises(TypeError, match="必须无参调用"):
        await redis_operation_with_fallback(_op, operation_name="bad_signature")  # pyright: ignore[reportArgumentType]


async def test_rejects_non_callable_operation():
    with pytest.raises(TypeError, match="必须是可调用对象"):
        await redis_operation_with_fallback("not-a-callable", operation_name="bad_type")  # pyright: ignore[reportArgumentType]


async def test_rejects_sync_return_value():
    """同步函数（返回值不可 await）同样必须显式失败，不得静默降级成默认值"""

    with pytest.raises(TypeError, match="必须返回 awaitable"):
        await redis_operation_with_fallback(lambda: "sync-result", operation_name="sync_op")  # pyright: ignore[reportArgumentType]


async def test_accepts_zero_arg_callable_returning_awaitable():
    """零参可调用对象返回 awaitable（cache.py 的 lambda 写法）必须照常放行"""

    async def _coro() -> str:
        return "ok"

    assert await redis_operation_with_fallback(lambda: _coro(), operation_name="lambda_op") == "ok"


async def test_zero_arg_async_function_runs():
    """标准写法（无参 async 函数）正常返回结果"""

    async def _op() -> int:
        return 7

    assert await redis_operation_with_fallback(_op, operation_name="async_op") == 7
