"""缓存服务（L1 本地 + L2 Redis 多级缓存）

支持：
- L1 本地缓存（TTLCache）：防热 key 击穿 Redis（进程级单例，跨请求共享）
- L2 Redis 缓存：分布式共享
- SingleFlight：防缓存击穿（热点 key 失效瞬间合并并发加载）
- 空值缓存：防缓存穿透
- Prometheus 指标：缓存命中率可观测

读流程：L1 -> L2 -> (SingleFlight 聚合) -> 返回并回填
写流程：先写 L2 -> 再写 L1（Cache-Aside Pattern）
"""
import json
import logging
from typing import Any, Awaitable, Callable, Optional, TypeVar

from redis.asyncio import Redis

from app.config import get_settings
from app.infrastructure.cache.local_cache import (NULL_VALUE_MARKER,
                                                  SingleFlight, TTLCache,
                                                  is_null_value)
from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback
from app.infrastructure.metrics.cache_metrics import (record_hit, record_loader,
                                                     record_miss)

logger = logging.getLogger(__name__)

T = TypeVar("T")

CACHE_TTL_HOUR = 3600
CACHE_TTL_DAY = 86400

# 进程级单例：L1 本地缓存和 SingleFlight 跨请求共享，否则防热 key 失效
_shared_l1: Optional[TTLCache] = None
_shared_singleflight: Optional[SingleFlight] = None


def _get_shared_l1() -> Optional[TTLCache]:
    """获取进程级共享 L1 缓存单例"""
    global _shared_l1
    if _shared_l1 is None:
        settings = get_settings()
        if settings.CACHE_L1_ENABLED:
            _shared_l1 = TTLCache(
                maxsize=settings.CACHE_L1_MAXSIZE,
                default_ttl=settings.CACHE_L1_TTL,
            )
    return _shared_l1


def _get_shared_singleflight() -> Optional[SingleFlight]:
    """获取进程级共享 SingleFlight 单例"""
    global _shared_singleflight
    if _shared_singleflight is None:
        settings = get_settings()
        if settings.CACHE_SINGLEFLIGHT_ENABLED:
            _shared_singleflight = SingleFlight()
    return _shared_singleflight


class CacheService:
    """多级缓存服务（L1 本地 + L2 Redis）

    集成 SingleFlight 防击穿、空值缓存防穿透。
    L1 缓存和 SingleFlight 为进程级单例，跨请求共享。
    """

    def __init__(self, redis: Redis):
        self.redis = redis
        settings = get_settings()
        # L1 本地缓存（进程级单例）
        self._l1_enabled = settings.CACHE_L1_ENABLED
        self._l1 = _get_shared_l1() if self._l1_enabled else None
        # SingleFlight（进程级单例）
        self._singleflight = _get_shared_singleflight() if settings.CACHE_SINGLEFLIGHT_ENABLED else None
        # 空值缓存
        self._null_enabled = settings.CACHE_NULL_ENABLED
        self._null_ttl = settings.CACHE_NULL_TTL

    async def get(
        self,
        key: str,
        default: Optional[T] = None,
    ) -> Optional[Any]:
        """多级缓存读取：L1 -> L2

        注意：本方法不触发回源加载。如需回源加载请使用 get_with_loader。
        """
        # 1. L1 本地缓存
        if self._l1 is not None:
            val = self._l1.get(key)
            if val is not None:
                # 检查空值标记
                if self._null_enabled and is_null_value(val):
                    record_hit("L1_null")
                    return default
                record_hit("L1")
                return val
            record_miss("L1")

        # 2. L2 Redis 缓存
        redis_val = await redis_operation_with_fallback(
            operation=lambda: self.redis.get(key),
            default=None,
            operation_name=f"cache_get:{key}",
        )
        if redis_val is not None:
            # 检查空值标记
            if self._null_enabled and is_null_value(redis_val):
                record_hit("L2_null")
                # 回填 L1
                if self._l1 is not None:
                    self._l1.set(key, redis_val, ttl=self._null_ttl)
                return default
            record_hit("L2")
            # 回填 L1
            if self._l1 is not None:
                self._l1.set(key, redis_val)
            return redis_val

        record_miss("L2")
        return default

    async def get_with_loader(
        self,
        key: str,
        loader: Callable[[], Awaitable[Optional[Any]]],
        ttl: int = CACHE_TTL_HOUR,
        default: Optional[T] = None,
    ) -> Optional[Any]:
        """带数据加载器的多级缓存读取

        当 L1 和 L2 都 miss 时，使用 SingleFlight 聚合并发请求，调用 loader 加载数据。
        loader 返回 None 时写入空值缓存（防穿透）。

        Args:
            key: 缓存 key
            loader: 数据加载函数（异步），返回 None 表示数据不存在
            ttl: 缓存 TTL（秒）
            default: 加载失败时的默认返回值
        """
        # 1. L1 本地缓存
        if self._l1 is not None:
            val = self._l1.get(key)
            if val is not None:
                if self._null_enabled and is_null_value(val):
                    record_hit("L1_null")
                    return default
                record_hit("L1")
                return val
            record_miss("L1")

        # 2. L2 Redis 缓存
        redis_val = await redis_operation_with_fallback(
            operation=lambda: self.redis.get(key),
            default=None,
            operation_name=f"cache_get_with_loader:{key}",
        )
        if redis_val is not None:
            if self._null_enabled and is_null_value(redis_val):
                record_hit("L2_null")
                if self._l1 is not None:
                    self._l1.set(key, redis_val, ttl=self._null_ttl)
                return default
            record_hit("L2")
            if self._l1 is not None:
                self._l1.set(key, redis_val)
            return redis_val

        record_miss("L2")

        # 3. SingleFlight 聚合回源加载
        async def _load() -> Optional[Any]:
            try:
                result = await loader()
            except Exception as e:
                record_loader("error")
                logger.error("缓存加载器执行失败 [%s]: %s", key, e)
                raise
            if result is None:
                # 数据不存在，写入空值缓存（防穿透）
                record_loader("miss")
                if self._null_enabled:
                    await self.set(key, NULL_VALUE_MARKER, ttl=self._null_ttl)
                return None
            record_loader("hit")
            # 写入 L2 + L1
            await self.set(key, result, ttl=ttl)
            return result

        if self._singleflight is not None:
            return await self._singleflight.do(key, _load)
        return await _load()

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = CACHE_TTL_HOUR,
    ) -> bool:
        """设置缓存（同时写入 L1 和 L2）"""
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)

        # 先写 L2
        ok = await redis_operation_with_fallback(
            operation=lambda: self.redis.setex(key, ttl, value),
            default=False,
            operation_name=f"cache_set:{key}",
        )

        # 再写 L1
        if self._l1 is not None:
            self._l1.set(key, value, ttl=ttl)
        return bool(ok)

    async def delete(self, key: str) -> bool:
        """删除缓存（先删 L2 再删 L1，Cache-Aside 模式）

        注意：redis.delete 返回被删除的 key 数量，key 不存在时返回 0。
        此处返回 True 表示删除操作已执行（不要求 key 必须存在），
        Redis 不可用时由 redis_operation_with_fallback 记录日志并返回默认值。
        """
        # 先删 L2
        await redis_operation_with_fallback(
            operation=lambda: self.redis.delete(key),
            default=0,
            operation_name=f"cache_delete:{key}",
        )
        # 再删 L1
        if self._l1 is not None:
            self._l1.delete(key)
        return True

    async def delete_pattern(self, pattern: str) -> int:
        """按通配符删除缓存（先删 L2 再删 L1）"""
        async def _delete_by_pattern() -> int:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
            if keys:
                return await self.redis.delete(*keys)
            return 0

        count = await redis_operation_with_fallback(
            operation=_delete_by_pattern,
            default=0,
            operation_name=f"cache_delete_pattern:{pattern}",
        )

        # 同步删除 L1 中匹配的 key
        if self._l1 is not None:
            self._l1.delete_pattern(pattern)

        return count

    async def get_json(
        self,
        key: str,
        default: Optional[T] = None,
    ) -> Optional[Any]:
        """获取 JSON 格式的缓存值"""
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
        """设置 JSON 格式的缓存值"""
        return await self.set(key, json.dumps(value, ensure_ascii=False), ttl)

    async def get_json_with_loader(
        self,
        key: str,
        loader: Callable[[], Awaitable[Optional[Any]]],
        ttl: int = CACHE_TTL_HOUR,
        default: Optional[T] = None,
    ) -> Optional[Any]:
        """带加载器的 JSON 缓存读取

        loader 返回的应该是可 JSON 序列化的 Python 对象（dict/list）。
        """
        # 1. 先尝试从缓存读取已序列化的 JSON
        if self._l1 is not None:
            val = self._l1.get(key)
            if val is not None:
                if self._null_enabled and is_null_value(val):
                    record_hit("L1_null")
                    return default
                try:
                    record_hit("L1")
                    return json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    record_miss("L1")
            else:
                record_miss("L1")

        redis_val = await redis_operation_with_fallback(
            operation=lambda: self.redis.get(key),
            default=None,
            operation_name=f"cache_get_json_with_loader:{key}",
        )
        if redis_val is not None:
            if self._null_enabled and is_null_value(redis_val):
                record_hit("L2_null")
                if self._l1 is not None:
                    self._l1.set(key, redis_val, ttl=self._null_ttl)
                return default
            try:
                result = json.loads(redis_val)
                record_hit("L2")
                if self._l1 is not None:
                    self._l1.set(key, redis_val)
                return result
            except (json.JSONDecodeError, TypeError):
                record_miss("L2")
        else:
            record_miss("L2")

        # 2. SingleFlight 聚合回源加载
        async def _load() -> Optional[Any]:
            try:
                result = await loader()
            except Exception as e:
                record_loader("error")
                logger.error("缓存加载器执行失败 [%s]: %s", key, e)
                raise
            if result is None:
                record_loader("miss")
                if self._null_enabled:
                    await self.set(key, NULL_VALUE_MARKER, ttl=self._null_ttl)
                return None
            record_loader("hit")
            # 序列化后写入 L2 + L1
            await self.set_json(key, result, ttl=ttl)
            return result

        if self._singleflight is not None:
            return await self._singleflight.do(key, _load)
        return await _load()


class DeptCacheKeys:
    TREE = "dept:tree"
    OPTIONS = "dept:options"

    @classmethod
    def all_patterns(cls) -> list[str]:
        return ["dept:tree*", "dept:options*"]
