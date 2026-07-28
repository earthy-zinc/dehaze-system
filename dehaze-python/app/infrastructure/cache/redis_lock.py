"""Redis 分布式锁

基于 SET NX EX + Lua 释放实现，保证释放操作的原子性（仅持锁者能释放）。
Redis 不可用时通过 redis_operation_with_fallback 降级（不阻塞业务主流程）。
"""
import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from app.dependencies.redis import get_redis_client
from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback

logger = logging.getLogger(__name__)

_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


class LockAcquireError(Exception):
    pass


async def acquire_lock(key: str, ttl_seconds: int, token: Optional[str] = None) -> Optional[str]:
    """尝试获取分布式锁，成功返回 token，失败返回 None"""
    token = token or secrets.token_hex(16)

    async def _acquire():
        redis = await get_redis_client()
        ok = await redis.set(key, token, nx=True, ex=ttl_seconds)
        return token if ok else None

    return await redis_operation_with_fallback(
        operation=_acquire,
        default=None,
        operation_name=f"lock_acquire:{key}",
    )


async def release_lock(key: str, token: str) -> bool:
    """释放锁，仅 token 匹配时成功"""
    async def _release():
        redis = await get_redis_client()
        result = await redis.eval(_RELEASE_LUA, 1, key, token)
        return bool(result)

    result = await redis_operation_with_fallback(
        operation=_release,
        default=False,
        operation_name=f"lock_release:{key}",
    )
    return bool(result)


@asynccontextmanager
async def distributed_lock(key: str, ttl_seconds: int) -> AsyncGenerator[Optional[str], None]:
    """分布式锁上下文管理器，获取失败时 yield None（业务可降级为无锁执行）"""
    token = await acquire_lock(key, ttl_seconds)
    try:
        yield token
    finally:
        if token:
            try:
                await release_lock(key, token)
            except Exception as e:
                logger.warning("释放分布式锁失败 key=%s: %s", key, e)


async def try_lock_or_raise(key: str, ttl_seconds: int, error_msg: str = "操作过于频繁，请稍后再试") -> str:
    """获取锁，失败抛出 LockAcquireError"""
    token = await acquire_lock(key, ttl_seconds)
    if token is None:
        raise LockAcquireError(error_msg)
    return token
