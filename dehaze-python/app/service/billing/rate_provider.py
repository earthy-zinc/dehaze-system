"""计费比例查询（RateProvider）

从 sys_ai_model 只读引用模型计费比例（input_rate/output_rate/cached_rate），
Redis 缓存 `ai:rate:{model_id}`（TTL 30 分钟）降低数据库压力。
支持降级模型比例查询（按实际使用的模型计费）。redis 经 get_redis_client() 自取。
"""

import json
import logging

from redis.exceptions import RedisError

from app.dependencies.redis import get_redis_client
from app.repository.ai_model_repository import ai_model_repository

logger = logging.getLogger(__name__)

RATE_CACHE_KEY = "ai:rate:{model_id}"
RATE_CACHE_TTL = 1800  # 30 分钟


class RateProvider:
    """模型计费比例查询（单例）"""

    @staticmethod
    async def get_rates(db, model_id: str) -> dict:
        """查询模型计费比例与输出上限，优先 Redis 缓存。

        返回 {input_rate, output_rate, cached_rate, max_output_tokens}，
        模型不存在或不可用时返回全 0，积分换算安全降级。
        """
        redis = await get_redis_client()
        try:
            cached = await redis.get(RATE_CACHE_KEY.format(model_id=model_id))
        except RedisError:
            logger.warning("费率缓存读取失败，降级查库 model_id=%s", model_id)
        else:
            if cached:
                try:
                    return json.loads(cached)
                except (ValueError, TypeError):
                    logger.warning("费率缓存数据异常，忽略并降级查库 model_id=%s", model_id)

        model = await ai_model_repository.get_by_model_id(db, model_id)
        if not model or model.status != 1:
            rates = {
                "input_rate": 0.0,
                "output_rate": 0.0,
                "cached_rate": 0.0,
                "max_output_tokens": 0,
            }
        else:
            rates = {
                "input_rate": float(model.input_rate),
                "output_rate": float(model.output_rate),
                "cached_rate": float(model.cached_rate),
                "max_output_tokens": model.max_output_tokens,
            }

        try:
            await redis.set(
                RATE_CACHE_KEY.format(model_id=model_id),
                json.dumps(rates),
                ex=RATE_CACHE_TTL,
            )
        except RedisError:
            logger.warning("费率缓存写入失败 model_id=%s", model_id)
        return rates

    @staticmethod
    async def calculate(
        db,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
    ) -> dict:
        """按模型计费比例换算积分。

        credits = input × input_rate + cached × cached_rate + output × output_rate
        credits_saved = cached × (input_rate - cached_rate)
        返回 {credits, credits_saved}
        """
        rates = await RateProvider.get_rates(db, model_id)
        input_rate = rates["input_rate"]
        output_rate = rates["output_rate"]
        cached_rate = rates["cached_rate"]

        credits = (
            input_tokens * input_rate + cached_tokens * cached_rate + output_tokens * output_rate
        )
        credits_saved = int(cached_tokens * (input_rate - cached_rate))
        return {
            "credits": int(credits),
            "credits_saved": credits_saved,
        }

