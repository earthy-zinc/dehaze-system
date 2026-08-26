"""预测结果缓存：(algorithmId, imageMd5) → Redis，24h TTL，带降级。"""

import json
import logging

from app.core.exceptions import BusinessException
from app.dependencies.redis import get_redis_client
from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback

logger = logging.getLogger(__name__)

PREDICTION_CACHE_TTL = 24 * 60 * 60


async def get_cached_prediction(cache_key: str) -> dict | None:
    """从 Redis 读取预测缓存（带降级）"""

    async def _get():
        redis = await get_redis_client()
        data = await redis.get(cache_key)
        if data:
            return json.loads(data)
        return None

    return await redis_operation_with_fallback(
        operation=_get,
        default=None,
        operation_name="prediction_cache_get",
    )


async def set_cached_prediction(cache_key: str, value: dict) -> None:
    """写入 Redis 预测缓存（带降级）"""

    async def _set():
        redis = await get_redis_client()
        await redis.setex(cache_key, PREDICTION_CACHE_TTL, json.dumps(value))

    await redis_operation_with_fallback(
        operation=_set,
        default=None,
        operation_name="prediction_cache_set",
    )


async def invalidate_prediction_cache(algorithm_id: int) -> int:
    """版本更新时失效该算法的所有预测缓存。

    返回实际删除的缓存键数量（0 表示本无 key 可删，属正常）；Redis 不可用导致
    删除失败时抛异常暴露，避免算法版本更新后仍命中旧预测脏数据。
    """

    async def _raise_failure():
        raise BusinessException(
            "预测缓存失效失败：Redis 不可用，算法版本更新后可能命中旧预测数据"
        )

    async def _invalidate():
        redis = await get_redis_client()
        pattern = f"prediction:{algorithm_id}:*"
        keys = []
        async for key in redis.scan_iter(match=pattern, count=100):
            keys.append(key)
        if keys:
            await redis.delete(*keys)
        return len(keys)

    return await redis_operation_with_fallback(
        operation=_invalidate,
        fallback=_raise_failure,
        operation_name="prediction_cache_invalidate",
    )
