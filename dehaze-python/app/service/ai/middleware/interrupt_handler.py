"""中断处理器

管理推理中断点与恢复，仅负责 Redis 存取，不涉及业务逻辑。

中断类型：confirm（用户确认）/ quota（配额不足）/ async_wait（异步任务等待）
存储键：ai:interrupt:{thread_id}，TTL 24 小时，resume 成功后清除。

confirm 有多个来源，恢复动作各不相同，故 interrupt.data 内带 confirmKind 子类型
（见 ConfirmKind），resume 据此路由；中断点只在恢复成功后清理，恢复失败须保留
以供重试。
"""

import json

from app.dependencies.redis import get_redis_client


class ConfirmKind:
    """confirm 中断的子类型（interrupt.data.confirmKind）。

    - ALGORITHM_RECOMMEND：算法推荐确认，接受时提交推荐正向反馈
    - TOOL_PERMISSION：工具权限确认，通过后在中间件重放该工具调用
    - DANGEROUS_OP：高风险操作确认——Shell/高风险代码执行（通过后在沙箱重放该次执行），
      以及并行子 Agent 写冲突（通过后覆盖写入，拒绝则放弃本次写入保留原结果）
    """

    ALGORITHM_RECOMMEND = "algorithm_recommend"
    TOOL_PERMISSION = "tool_permission"
    DANGEROUS_OP = "dangerous_op"


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
