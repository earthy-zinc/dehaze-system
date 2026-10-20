"""配额召回组件（QuotaRecall）：并行子 Agent 派发前预扣 + 执行中温和停止

设计文档 §7.2 配额召回机制：子智能体派发前主 Agent 预扣本步预估消耗；并行执行
过程中配额不足时，未启动的子任务召回不执行、正在执行的等待当前步完成后停止、
已完成的保留，并推送"部分子任务因配额不足未执行"提示。

由 paradigms/ 并行执行器集成调用（经共享 ctx 与契约接口交互，不触碰计费实现）。

预扣为批次预留：批结束经 settle_batch 全额退回配额（本批实际消耗已汇总进最终
usage，由 after_agent 主结算统一扣减，不退会与主结算重复扣减）。
"""

import logging

from app.database import get_db_session
from app.infrastructure.sse.sse_emitter_manager import sse_emitter_manager
from app.service.billing.estimate_service import estimate_service
from app.service.billing.quota_service import quota_service
from app.service.billing.rate_provider import rate_provider

logger = logging.getLogger(__name__)

_SKIPPED_STATUS = 3


async def _actual_credits(db, model_id: str, usage: dict) -> int | None:
    """按模型售价换算实际消耗积分；售价未配置/换算失败返回 None（调用方按预扣计）。"""
    try:
        calc = await rate_provider.calculate(
            db,
            model_id,
            None,
            int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
            int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
            int(usage.get("cached_input_tokens") or 0),
        )
    except Exception as e:
        logger.warning("批量实际消耗换算失败: %s", e)
        return None
    return int(calc["credits"])


class QuotaRecall:
    """并行批次配额预扣与召回控制"""

    async def precharge_batch(self, ctx: dict, tasks: list[dict]) -> int | None:
        """派发前按子任务说明预估预扣本批次消耗。

        预估按子任务描述长度逐条累加（子任务说明即该次 LLM 调用的输入主体），
        避免按空上下文低估导致预留不足。

        Returns:
            预扣积分数（无计费上下文或非 AI 计费场景为 0）；
            None 表示预算/配额不足，调用方应放弃本批并降级处理。
        """
        if not tasks:
            return 0
        bc = ctx.get("billing_context")
        if not bc or not bc.get("user_id"):
            # 无计费上下文（非 AI 计费场景）不阻断批量执行
            return 0
        try:
            async with get_db_session() as db:
                total = 0
                for task in tasks:
                    total += await estimate_service.estimate_step_credits(
                        db,
                        ctx.get("model_id") or "",
                        [{"content": task.get("description") or ""}],
                    )
            remaining = int(bc.get("remaining_budget", 0))
            if total > remaining:
                return None
            async with get_db_session() as db:
                if not await quota_service.pre_deduct(db, bc["user_id"], total):
                    return None
            bc["remaining_budget"] = remaining - total
            bc["precharged_batch"] = int(bc.get("precharged_batch", 0)) + total
            return total
        except Exception as e:
            logger.warning("批量预扣失败: %s", e, exc_info=True)
            return None

    async def settle_batch(self, ctx: dict, reserved: int, usage: dict) -> None:
        """批结束结算：退回本批预留并按实际消耗修正剩余预算。

        失败/跳过任务的预留在此一并退回（实际消耗为零，配额侧不留痕）。
        """
        bc = ctx.get("billing_context")
        if reserved <= 0 or not bc or not bc.get("user_id"):
            return
        try:
            async with get_db_session() as db:
                actual = await _actual_credits(db, ctx.get("model_id") or "", usage)
                await quota_service.refund(bc["user_id"], reserved)
        except Exception as e:
            logger.warning("批量预扣结算失败: %s", e, exc_info=True)
            return
        # 预估偏保守会提前耗尽预算：实际低于预估的部分回补预算池
        settled = reserved if actual is None else actual
        bc["remaining_budget"] = int(bc.get("remaining_budget", 0)) + reserved - settled
        bc["precharged_batch"] = int(bc.get("precharged_batch", 0)) - reserved

    def check_and_recall(self, ctx: dict, pending: list) -> list:
        """执行中配额不足时召回未启动的子任务（纯 ctx 判定，无 IO）。

        Args:
            ctx: 共享运行时上下文。
            pending: 尚未启动的子任务列表。

        Returns:
            被召回的 pending 子任务列表（调用方应取消其执行并记录为 skipped）。
            召回提示由调用方在临界区外经 notify_partial_skipped 推送。
        """
        if self._budget_exhausted(ctx) and pending:
            return list(pending)
        return []

    async def notify_partial_skipped(self, ctx: dict, count: int) -> None:
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


quota_recall = QuotaRecall()
