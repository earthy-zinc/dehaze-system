"""L1 本地缓存 + SingleFlight + 空值缓存实现

基于 Python 标准库实现，避免引入额外依赖：
- TTLCache：基于 OrderedDict + threading.Lock，支持 TTL 过期
- SingleFlight：基于 asyncio.Lock + dict，合并并发加载请求
- NULL_VALUE_MARKER：空值标记，防缓存穿透
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from threading import Lock
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 空值缓存标记（与 Go 端 NullCache.NullValueMarker 保持一致）
NULL_VALUE_MARKER = "__NULL__"


class TTLCache:
    """线程安全的 TTL 本地缓存（L1）

    基于 OrderedDict 实现 LRU 淘汰 + TTL 过期。
    适用于多 worker 进程内的热 key 缓存。
    """

    def __init__(self, maxsize: int = 1000, default_ttl: int = 300):
        self._maxsize = maxsize
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        """获取缓存值，未命中或已过期返回 None"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            value, expire_at = entry
            if time.monotonic() >= expire_at:
                # 已过期，删除
                self._cache.pop(key, None)
                return None
            # LRU：移到末尾
            self._cache.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """设置缓存值"""
        expire_seconds = ttl if ttl is not None and ttl > 0 else self._default_ttl
        expire_at = time.monotonic() + expire_seconds
        with self._lock:
            # 已存在则先删除（避免 OrderedDict 重复 key）
            if key in self._cache:
                self._cache.pop(key)
            self._cache[key] = (value, expire_at)
            # LRU 淘汰
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def delete(self, key: str) -> bool:
        """删除缓存值，返回是否删除成功"""
        with self._lock:
            return self._cache.pop(key, None) is not None

    def delete_pattern(self, pattern: str) -> int:
        """按通配符删除缓存，返回删除数量"""
        # 简单实现：支持 * 通配符
        import fnmatch

        with self._lock:
            keys_to_delete = [k for k in self._cache if fnmatch.fnmatch(k, pattern)]
            for k in keys_to_delete:
                self._cache.pop(k, None)
            return len(keys_to_delete)

    def clear(self) -> None:
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """返回当前缓存条目数"""
        with self._lock:
            return len(self._cache)


class SingleFlight:
    """异步 SingleFlight（防缓存击穿）

    相同 key 的并发请求，第一个执行加载函数，其余等待结果共享。
    基于 asyncio.Lock + dict 实现。
    """

    def __init__(self):
        self._calls: dict[str, asyncio.Future[Any]] = {}
        self._lock = asyncio.Lock()

    async def do(
        self,
        key: str,
        fn: Callable[[], Awaitable[T]],
    ) -> T:
        """执行加载函数，相同 key 的并发请求合并

        Args:
            key: 缓存 key
            fn: 加载函数（异步）

        Returns:
            加载结果
        """
        async with self._lock:
            existing = self._calls.get(key)
            if existing is not None:
                logger.debug("SingleFlight 命中合并请求: %s", key)
                # 释放锁等待结果
                pass
            else:
                # 创建 Future 并注册
                future: asyncio.Future[T] = asyncio.get_event_loop().create_future()
                self._calls[key] = future
                # 启动加载任务
                asyncio.create_task(self._load(key, fn, future))
                existing = future

        # 等待结果（锁已释放）
        return await existing

    async def _load(
        self,
        key: str,
        fn: Callable[[], Awaitable[T]],
        future: asyncio.Future[T],
    ) -> None:
        """执行加载函数并设置 Future 结果"""
        try:
            result = await fn()
            if not future.done():
                future.set_result(result)
        except Exception as e:
            if not future.done():
                future.set_exception(e)
        finally:
            # 清理调用记录
            async with self._lock:
                self._calls.pop(key, None)

    def forget(self, key: str) -> None:
        """清除指定 key 的 in-flight 记录"""
        self._calls.pop(key, None)


def is_null_value(value: Any) -> bool:
    """判断是否为空值缓存标记"""
    return value == NULL_VALUE_MARKER
