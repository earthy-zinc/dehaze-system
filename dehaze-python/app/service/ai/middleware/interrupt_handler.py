"""中断处理器

管理推理中断点与恢复，仅负责 Redis 存取，不涉及业务逻辑。

中断类型：confirm（用户确认）/ quota（配额不足）/ async_wait（异步任务等待）
存储键：ai:interrupt:{thread_id}，TTL 24 小时，resume 时清除。
"""

import json

from app.dependencies.redis import get_redis_client


class InterruptHandler:
    """推理中断处理器"""

    INTERRUPT_KEY = "ai:interrupt:{thread_id}"
    INTERRUPT_TTL = 86400  # 24 小时

    async def save_interrupt(self, thread_id: str, interrupt_type: str, data: dict) -> None:
        """保存中断点信息到 Redis"""
        redis = await get_redis_client()
        await redis.set(
            self.INTERRUPT_KEY.format(thread_id=thread_id),
            json.dumps({"type": interrupt_type, "data": data}),
            ex=self.INTERRUPT_TTL,
        )

    async def get_interrupt(self, thread_id: str) -> dict | None:
        """获取中断点信息"""
        redis = await get_redis_client()
        raw = await redis.get(self.INTERRUPT_KEY.format(thread_id=thread_id))
        if raw:
            return json.loads(raw)
        return None

    async def clear_interrupt(self, thread_id: str) -> None:
        """清除中断点"""
        redis = await get_redis_client()
        await redis.delete(self.INTERRUPT_KEY.format(thread_id=thread_id))


interrupt_handler = InterruptHandler()
