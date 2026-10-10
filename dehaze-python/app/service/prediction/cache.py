"""图像任务结果缓存：(model_id, imageMd5) → Redis，24h TTL，带降级。"""

import json
import logging

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
