"""配额召回组件（QuotaRecall）：并行子 Agent 派发前预扣 + 执行中温和停止

设计文档 §7.2 配额召回机制：子智能体派发前主 Agent 预扣本步预估消耗；并行执行
过程中配额不足时，未启动的子任务召回不执行、正在执行的等待当前步完成后停止、
已完成的保留，并推送"部分子任务因配额不足未执行"提示。

由 paradigms/ 并行执行器集成调用（经共享 ctx 与契约接口交互，不触碰计费实现）。
"""

import logging

from app.database import get_db_session
from app.infrastructure.sse.sse_emitter_manager import sse_emitter_manager
from app.service.billing.estimate_service import EstimateService
from app.service.billing.quota_service import QuotaService

logger = logging.getLogger(__name__)

_SKIPPED_STATUS = 3


class QuotaRecall:
    """并行批次配额预扣与召回控制"""

    async def precharge_batch(self, ctx: dict, batch_size: int) -> bool:
        """派发前按子任务数预估预扣本批次消耗。

        从共享 ctx 的 billing_context 读取预算池，批量预估超出剩余预算则返回 False
        （调用方应放弃本批预扣并降级处理），否则扣减剩余预算返回 True。
        """
        if batch_size <= 0:
            return True
        bc = ctx.get("billing_context")
        if not bc or not bc.get("user_id"):
            # 无计费上下文（非 AI 计费场景）不阻断批量执行
            return True
        try:
            async with get_db_session() as db:
                per_subtask = await EstimateService.estimate_step_credits(
                    db,
                    ctx.get("model_id") or "",
                    ctx.get("messages") or [],
                )
            total = per_subtask * batch_size
            remaining = int(bc.get("remaining_budget", 0))
            if total > remaining:
                return False
            # 预扣：配额侧扣减 + 预算池扣减
            async with get_db_session() as db:
                if not await QuotaService.pre_deduct(db, bc["user_id"], total):
                    return False
            bc["remaining_budget"] = remaining - total
            bc.setdefault("precharged_batch", 0)
            bc["precharged_batch"] += total
            return True
        except Exception as e:
            logger.warning("批量预扣失败: %s", e, exc_info=True)
            return False

    async def check_and_recall(self, ctx: dict, running: list, pending: list) -> list:
        """执行中配额不足时召回未启动的子任务。

        Args:
            ctx: 共享运行时上下文。
            running: 正在执行的子任务列表（已完成/进行中）。
            pending: 尚未启动的子任务列表。

        Returns:
            被召回的 pending 子任务列表（调用方应取消其执行并记录为 skipped）。
        """
        recalled: list = []
        if self._budget_exhausted(ctx) and pending:
            recalled = list(pending)
            await self._notify_partial_skipped(ctx, len(recalled))
        return recalled

    @staticmethod
    def _budget_exhausted(ctx: dict) -> bool:
        """判定配额/预算是否耗尽：Token 预算或计费预算任一超限即认为不足。"""
        token_used = ctx.get("token_used", 0)
        token_budget = ctx.get("token_budget", 0)
        if token_budget and token_used >= token_budget:
            return True
        bc = ctx.get("billing_context")
        if bc and "remaining_budget" in bc:
            return int(bc.get("remaining_budget", 0)) <= 0
        return False

    async def _notify_partial_skipped(self, ctx: dict, count: int) -> None:
        """推送"部分子任务因配额不足未执行"的 thought 事件（status=3 跳过通道）。"""
        stream_session_id = ctx.get("stream_session_id")
        if not stream_session_id:
            return
        try:
            await sse_emitter_manager.send_event(
                stream_session_id,
                "thought",
                {
                    "position": 0,
                    "thought": f"部分子任务因配额不足未执行（{count} 项）",
                    "tool": "quota_recall",
                    "toolInput": {"recalled": count},
                    "observation": "配额不足，未启动的子任务已跳过",
                    "status": _SKIPPED_STATUS,
                    "latencyMs": 0,
                },
            )
        except Exception:
            logger.warning("配额召回跳过提示推送失败", exc_info=True)


quota_recall = QuotaRecall()
