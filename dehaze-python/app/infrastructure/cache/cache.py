import json
import logging
from typing import Any, Optional, TypeVar

from redis.asyncio import Redis

from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback

logger = logging.getLogger(__name__)

T = TypeVar("T")

CACHE_TTL_HOUR = 3600
CACHE_TTL_DAY = 86400


class CacheService:
    def __init__(self, redis: Redis):
        self.redis = redis

    async def get(
        self,
        key: str,
        default: Optional[T] = None,
    ) -> Optional[Any]:
        return await redis_operation_with_fallback(
            operation=lambda: self.redis.get(key),
            default=default,
            operation_name=f"cache_get:{key}",
        )

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = CACHE_TTL_HOUR,
    ) -> bool:
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)

        return await redis_operation_with_fallback(
            operation=lambda: self.redis.setex(key, ttl, value),
            default=False,
            operation_name=f"cache_set:{key}",
        ) or False

    async def delete(self, key: str) -> bool:
        return await redis_operation_with_fallback(
            operation=lambda: self.redis.delete(key),
            default=False,
            operation_name=f"cache_delete:{key}",
        ) or False

    async def delete_pattern(self, pattern: str) -> int:
        async def _delete_by_pattern() -> int:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
            if keys:
                return await self.redis.delete(*keys)
            return 0

        return await redis_operation_with_fallback(
            operation=_delete_by_pattern,
            default=0,
            operation_name=f"cache_delete_pattern:{pattern}",
        ) or 0

    async def get_json(
        self,
        key: str,
        default: Optional[T] = None,
    ) -> Optional[Any]:
        value = await self.get(key)
        if value is None:
            return default
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default

    async def set_json(
        self,
        key: str,
        value: Any,
        ttl: int = CACHE_TTL_HOUR,
    ) -> bool:
        return await self.set(key, json.dumps(value, ensure_ascii=False), ttl)


class DeptCacheKeys:
    TREE = "dept:tree"
    OPTIONS = "dept:options"

    @classmethod
    def all_patterns(cls) -> list[str]:
        return ["dept:tree*", "dept:options*"]
