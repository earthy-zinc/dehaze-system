"""积分预估（EstimateService）

基于历史上下文均值 + 当前消息长度，预估本次对话的积分消耗。
预扣减值 = 预估积分（见 AI计费管理后端实现 §4.2）。

redis 经 get_redis_client() 自取。
"""

from redis.exceptions import RedisError

from app.config import settings
from app.dependencies.redis import get_redis_client
from app.service.billing.rate_provider import rate_provider

CTX_AVG_KEY = "ai:ctx:avg:{conversation_id}"
CTX_AVG_TTL = 600  # 10 分钟


class EstimateService:
    """积分预估（rate_provider 构造注入，默认模块单例）"""

    def __init__(self, rate_provider=rate_provider):
        self.rate_provider = rate_provider

    async def estimate_credits(self, 
        db,
        user_id: int,
        conversation_id: int,
        content: str,
        model_id: str,
    ) -> int:
        """预估本次对话积分消耗并更新历史上下文均值，返回预估积分。"""
        redis = await get_redis_client()
        rates = await self.rate_provider.get_rates(db, model_id)
        input_rate = rates["input_rate"]
        output_rate = rates["output_rate"]

        # 历史上下文 Token：Redis 均值 × 0.8
        ctx_avg = await self._read_ctx_avg(redis, conversation_id)
        context_tokens = ctx_avg * 0.8

        # 当前消息 Token × 保守系数
        msg_tokens = (len(content) // 4) * settings.AI_BILLING_ESTIMATE_INPUT_FACTOR

        # 记忆注入 + 工具定义
        input_estimate = (
            context_tokens
            + msg_tokens
            + settings.AI_BILLING_ESTIMATE_MEMORY_TOKENS
            + settings.AI_BILLING_ESTIMATE_TOOL_TOKENS
        )

        # 预估输出 Token
        output_estimate = rates["max_output_tokens"] * settings.AI_BILLING_ESTIMATE_OUTPUT_FACTOR

        estimated = int(input_estimate * input_rate + output_estimate * output_rate)

        await self._update_ctx_avg(redis, conversation_id, ctx_avg, int(input_estimate))
        return estimated

    async def estimate_step_credits(self, 
        db,
        model_id: str,
        messages: list[dict],
    ) -> int:
        """预估多步推理中单步的积分消耗（滚动预算校验用）"""
        rates = await self.rate_provider.get_rates(db, model_id)
        ctx_tokens = sum(len(msg.get("content") or "") // 4 for msg in messages)
        output_estimate = rates["max_output_tokens"] * settings.AI_BILLING_ESTIMATE_OUTPUT_FACTOR
        return int(ctx_tokens * rates["input_rate"] + output_estimate * rates["output_rate"])

    async def _read_ctx_avg(self, redis, conversation_id: int) -> int:
        """读取会话历史上下文 Token 均值，Redis 不可用或缺省时返回 0"""
        if not redis:
            return 0
        try:
            raw = await redis.get(CTX_AVG_KEY.format(conversation_id=conversation_id))
            return int(raw) if raw else 0
        except (RedisError, ValueError, TypeError):
            # 上下文均值读取失败/损坏按无历史处理（尽力而为估算）
            return 0

    async def _update_ctx_avg(self, 
        redis,
        conversation_id: int,
        old_avg: int,
        new_sample: int,
    ) -> None:
        """滑动平均更新会话上下文 Token 均值"""
        if not redis:
            return
        weight = settings.AI_BILLING_CTX_AVG_WEIGHT
        new_avg = int(old_avg * (1 - weight) + new_sample * weight)
        try:
            await redis.set(
                CTX_AVG_KEY.format(conversation_id=conversation_id),
                new_avg,
                ex=CTX_AVG_TTL,
            )
        except RedisError:
            # 上下文均值写入失败不影响本次预估（尽力而为）
            pass



estimate_service = EstimateService()
