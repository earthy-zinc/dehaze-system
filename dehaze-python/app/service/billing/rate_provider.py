"""计费比例查询（RateProvider）

从 sys_ai_model_price 只读引用模型用户售价（价格版本基础档位），
Redis 缓存 `ai:rate:{model_id}`（TTL 30 分钟）降低数据库压力。
支持降级模型比例查询（按实际使用的模型计费）。redis 经 get_redis_client() 自取。
"""

import json
import logging
from datetime import datetime

from redis.exceptions import RedisError

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.timezone import is_peak_hour
from app.dependencies.redis import get_redis_client
from app.repository.ai_model_price_repository import ai_model_price_repository
from app.repository.ai_model_repository import ai_model_repository
from app.service.ai_model_price_service import ai_model_price_service

logger = logging.getLogger(__name__)

RATE_CACHE_KEY = "ai:rate:{model_id}"
RATE_CACHE_TTL = 1800  # 30 分钟


class RateProvider:
    """模型计费比例查询（仓储/服务构造注入，默认模块单例）"""

    def __init__(
        self,
        ai_model_repository=ai_model_repository,
        ai_model_price_repository=ai_model_price_repository,
        ai_model_price_service=ai_model_price_service,
    ):
        self.ai_model_repository = ai_model_repository
        self.ai_model_price_repository = ai_model_price_repository
        self.ai_model_price_service = ai_model_price_service

    async def get_rates(self, db, model_id: str) -> dict:
        """查询模型基础计费比例与输出上限，优先 Redis 缓存。

        价格来源：sys_ai_model_price 生效版本的基础档位（min_tokens 最小的档位行），
        按当前时刻时段（peak/idle）取 input/output/cached 单价，换算为每 token 积分
        （unit_price 积分/百万token ÷ 1e6）。模型不存在/不可用时返回全 0，积分换算安全降级。
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

        model = await self.ai_model_repository.get_by_model_id(db, model_id)
        if not model or model.status != 1:
            rates = {
                "input_rate": 0.0,
                "output_rate": 0.0,
                "cached_rate": 0.0,
                "max_output_tokens": 0,
            }
        else:
            prices = await self._base_unit_prices(db, model_id, model.provider_id)
            rates = {
                "input_rate": prices["input"] / 1_000_000,
                "output_rate": prices["output"] / 1_000_000,
                "cached_rate": prices["cached"] / 1_000_000,
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

    async def _base_unit_prices(self, db, model_id: str, provider_id: int) -> dict:
        """取生效价格版本中 min_tokens 最小的 input/output/cached 档位单价（当前时段，积分/百万token）"""
        version = await self.ai_model_price_repository.get_effective_version(
            db, model_id, provider_id, datetime.now()
        )
        if version is None:
            return {"input": 0.0, "output": 0.0, "cached": 0.0}
        details = await self.ai_model_price_repository.list_details(db, version.id)
        slot = "peak" if is_peak_hour(datetime.now()) else "idle"

        def _base(token_type: str) -> float:
            rows = [d for d in details if d.token_type == token_type and d.time_slot == slot]
            if not rows:
                return 0.0
            return float(min(rows, key=lambda d: (d.min_tokens, d.id)).unit_price)

        return {"input": _base("input"), "output": _base("output"), "cached": _base("cached")}

    async def calculate(
        self,
        db,
        model_id: str,
        provider_id: int | None,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
    ) -> dict:
        """按模型用户售价换算积分（三维档位匹配，见 AI模型管理 §2.12）。

        返回 {credits, credits_saved}；模型未配置售价时抛错暴露（不做静默 0 兜底）。
        """
        result = await self.ai_model_price_service.calculate(
            db,
            model_id,
            provider_id,
            datetime.now(),
            input_tokens,
            cached_tokens,
            output_tokens,
        )
        if not result["configured"]:
            raise BusinessException(
                ResultCode.AI_MODEL_NOT_AVAILABLE, f"模型 {model_id} 未配置用户售价"
            )
        return result


rate_provider = RateProvider()
