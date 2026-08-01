"""缓存服务（L1 本地 + L2 Redis 多级缓存）

支持：
- L1 本地缓存（TTLCache）：防热 key 击穿 Redis（进程级单例，跨请求共享）
- L2 Redis 缓存：分布式共享
- SingleFlight：防缓存击穿（热点 key 失效瞬间合并并发加载）
- 空值缓存：防缓存穿透
- Prometheus 指标：缓存命中率可观测
- 多实例缓存失效广播：通过 Redis Pub/Sub 同步 L1 缓存失效

读流程：L1 -> L2 -> (SingleFlight 聚合) -> 返回并回填
写流程：先写 L2 -> 再写 L1（Cache-Aside Pattern）
"""
import asyncio
import json
import logging
import uuid
from typing import Any, Awaitable, Callable, Optional, TypeVar

from redis.asyncio import Redis

from app.config import settings
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

# 本实例标识，用于 Pub/Sub 防自消费
_INSTANCE_ID = str(uuid.uuid4())

# Pub/Sub 监听任务
_pubsub_task: Optional[asyncio.Task] = None


def _get_shared_l1() -> Optional[TTLCache]:
    """获取进程级共享 L1 缓存单例"""
    global _shared_l1
    if _shared_l1 is None:
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
        同时发布 Pub/Sub 失效消息，通知其他实例清除 L1 缓存。
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
        # 广播失效消息，通知其他实例清除 L1
        await _publish_invalidation("key", key)
        return True

    async def delete_pattern(self, pattern: str) -> int:
        """按通配符删除缓存（先删 L2 再删 L1）

        同时发布 Pub/Sub 失效消息，通知其他实例按 pattern 清除 L1 缓存。
        """
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

        # 广播失效消息，通知其他实例清除 L1
        await _publish_invalidation("pattern", pattern)

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


async def _publish_invalidation(msg_type: str, key: str) -> None:
    """发布缓存失效消息到 Pub/Sub 频道，通知其他实例清除 L1 缓存。

    Args:
        msg_type: 消息类型，"key"（单个 key）或 "pattern"（通配符）
        key: 缓存 key 或 pattern
    """
    payload = json.dumps({
        "type": msg_type,
        "key": key,
        "senderId": _INSTANCE_ID,
    })

    async def _publish():
        from app.dependencies.redis import get_redis_client
        redis = await get_redis_client()
        await redis.publish(settings.CACHE_INVALIDATION_CHANNEL, payload)

    await redis_operation_with_fallback(
        operation=_publish,
        default=None,
        operation_name=f"cache_invalidation_publish:{msg_type}:{key}",
    )


async def start_cache_invalidation_listener() -> None:
    """启动缓存失效广播订阅。

    在应用启动时调用（lifespan），订阅 CACHE_INVALIDATION_CHANNEL 频道，
    收到其他实例发布的失效消息时清除本地 L1 缓存。
    忽略自己发送的消息（通过 senderId 判断）。
    """
    global _pubsub_task
    if _pubsub_task is not None:
        return

    if not settings.CACHE_L1_ENABLED:
        logger.debug("L1 缓存未启用，跳过缓存失效广播订阅")
        return

    _pubsub_task = asyncio.create_task(_subscription_loop())
    logger.debug(
        "缓存失效广播订阅已启动: channel=%s, instanceId=%s",
        settings.CACHE_INVALIDATION_CHANNEL, _INSTANCE_ID,
    )


async def _subscription_loop() -> None:
    """订阅缓存失效频道的循环任务，断开时自动重连。"""
    channel = settings.CACHE_INVALIDATION_CHANNEL

    while True:
        try:
            from app.dependencies.redis import get_redis_client
            redis = await get_redis_client()
            pubsub = redis.pubsub()
            await pubsub.subscribe(channel)
            logger.debug("已订阅缓存失效频道: %s", channel)

            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                await _handle_invalidation_message(message["data"])

            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
        except asyncio.CancelledError:
            logger.debug("缓存失效广播订阅任务已取消")
            break
        except Exception as e:
            logger.error("缓存失效 Pub/Sub 异常: %s, 3秒后重连", e, exc_info=True)
            await asyncio.sleep(3)


async def _handle_invalidation_message(data: str) -> None:
    """处理收到的缓存失效消息，清除本地 L1 缓存。

    忽略自己发送的消息（通过 senderId 判断）。
    """
    try:
        msg = json.loads(data)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("缓存失效消息解析失败: %s", e)
        return

    sender_id = msg.get("senderId")
    if sender_id == _INSTANCE_ID:
        return

    msg_type = msg.get("type")
    key = msg.get("key")
    if not key:
        return

    l1 = _get_shared_l1()
    if l1 is None:
        return

    if msg_type == "key":
        l1.delete(key)
    elif msg_type == "pattern":
        l1.delete_pattern(key)
    else:
        logger.warning("未知的缓存失效消息类型: %s", msg_type)
        return

    logger.debug(
        "收到缓存失效消息并清除本地 L1: type=%s, key=%s, from=%s",
        msg_type, key, sender_id,
    )


async def stop_cache_invalidation_listener() -> None:
    """停止缓存失效广播订阅。

    在应用关闭时调用（lifespan）。
    """
    global _pubsub_task
    if _pubsub_task is None:
        return

    _pubsub_task.cancel()
    try:
        await _pubsub_task
    except asyncio.CancelledError:
        pass
    _pubsub_task = None
    logger.debug("缓存失效广播订阅已停止")
