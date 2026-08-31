"""计费总入口（BillingService）

计量 → 预扣 → 实扣 → 记录，贯穿 Agent 推理全生命周期：
- pre_charge    （before_agent 钩子）：欠费熔断 + 预估 + 配额/余额预扣 + 建计费记录
- check_budget  （before_model 钩子）：滚动预算校验，不足中断
- settle        （after_agent 钩子）：按实际用量差额退补 + 更新计费记录 + 流水
- record_tool_llm / record_kb_inject：工具推理与知识库注入的独立计费记录
"""

import asyncio
import logging
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db_session
from app.repository.ai_billing_repository import ai_billing_repository
from app.repository.ai_credit_log_repository import ai_credit_log_repository
from app.service.billing.balance_service import balance_service
from app.service.billing.billing_anomaly_service import billing_anomaly_service
from app.service.billing.cost_service import cost_service
from app.service.billing.estimate_service import estimate_service
from app.service.billing.quota_service import quota_service
from app.service.billing.rate_provider import rate_provider
from app.service.member.growth_service import member_growth_service

logger = logging.getLogger(__name__)

# 异步后台任务引用，防止被垃圾回收
_pending_tasks: set[asyncio.Task] = set()


def _publish_chat_completed(user_id: int) -> None:
    """发布 ai.chat.completed 事件（fire-and-forget）：会员模块消费后写 AI 使用激励成长值。

    一次对话完成仅发布一次（多步推理/子智能体的多次 LLM 调用不重复），
    消费失败仅记 warning，不影响计费主流程。
    """

    async def _run() -> None:
        try:
            async with get_db_session() as db:
                await member_growth_service.add_behavior_growth(db, user_id, "ai_consume")
        except Exception:
            logger.warning("ai.chat.completed 消费失败（AI 使用激励） user_id=%s", user_id, exc_info=True)

    task = asyncio.create_task(_run())
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)


class BillingService:
    """计费总入口：计量 → 预扣 → 实扣 → 记录"""

    def __init__(
        self,
        ai_billing_repository=ai_billing_repository,
        ai_credit_log_repository=ai_credit_log_repository,
    ):
        self.ai_billing_repository = ai_billing_repository
        self.ai_credit_log_repository = ai_credit_log_repository

    # ── 预校验 + 预扣 ──────────────────────────────

    async def pre_charge(self, 
        db: AsyncSession,
        user_id: int,
        conversation_id: int,
        message_id: int,
        content: str,
        model_id: str,
    ) -> dict:
        """预校验 + 预扣减（before_agent 钩子调用）。

        返回 PreDeductContext（含 billing_id）表示预扣成功；
        配额/余额/欠费任一不满足时返回中断数据（含 stop_reason）供钩子阻断推理。
        """
        # 1. 欠费熔断
        if await balance_service.is_arrears(user_id):
            return {
                "final_response": "账户欠费，请充值后继续使用",
                "stop_reason": "arrears",
            }

        # 2. 预估
        estimated = await estimate_service.estimate_credits(
            db, user_id, conversation_id, content, model_id
        )

        # 3. 配额校验（失败计入连续配额不足计数，达阈值告警，见后端实现 §4.7）
        if not await quota_service.check_quota(db, user_id, estimated):
            await billing_anomaly_service.record_quota_fail(db, user_id)
            return {
                "final_response": "今日或本月 AI 积分配额不足，请升级会员或明日再试",
                "stop_reason": "quota_exceeded",
            }

        # 4. 余额校验
        if not await balance_service.check_balance(db, user_id, estimated):
            return {
                "final_response": "积分余额不足，请充值后继续使用",
                "stop_reason": "balance_exceeded",
            }

        # 5. 配额预扣（失败已整体回滚，无副作用；并发窗口内失败同样计入配额不足计数）
        if not await quota_service.pre_deduct(db, user_id, estimated):
            await billing_anomaly_service.record_quota_fail(db, user_id)
            return {
                "final_response": "今日或本月 AI 积分配额不足，请升级会员或明日再试",
                "stop_reason": "quota_exceeded",
            }

        # 6. 余额预扣（失败回滚已预扣配额）
        if not await balance_service.pre_deduct(db, user_id, estimated):
            await quota_service.refund(user_id, estimated)
            return {
                "final_response": "积分余额不足，请充值后继续使用",
                "stop_reason": "balance_exceeded",
            }

        # 7. 创建计费记录（预扣值，其余字段待实扣结算更新）
        billing = await self.ai_billing_repository.create_billing(
            db,
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            model=model_id,
            bill_type="chat",
            pre_deduct=estimated,
        )

        # 8. 组装 PreDeductContext（滚动预算基于预扣总预算）
        return {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "estimated_credits": estimated,
            "budget_pool": estimated,
            "remaining_budget": estimated,
            "billing_id": billing.id,
        }

    # ── 滚动预算校验 ──────────────────────────────

    async def check_budget(self, state: dict, step_estimated: int) -> dict | None:
        """滚动预算校验（before_model 钩子调用）。

        remaining_budget < 单步预估 → 返回中断数据（type=quota）。
        """
        bc = state.get("billing_context")
        if not bc:
            return None
        if step_estimated > bc["remaining_budget"]:
            return {
                "final_response": "本次推理积分预算不足，已暂停，请升级会员后重试",
                "stop_reason": "quota_exceeded",
                "interrupt": {"type": "quota"},
            }
        return None

    # ── 实扣结算 ──────────────────────────────────

    async def settle(self, 
        db: AsyncSession,
        user_id: int,
        conversation_id: int,
        message_id: int,
        model_id: str,
        actual_model_id: str | None,
        usage: dict,
        bill_type: str = "chat",
        *,
        adjustment: bool = False,
        request_id: str | None = None,
        provider_id: int | None = None,
        error_code: str | None = None,
        latency_ms: int | None = None,
    ) -> dict:
        """实扣结算（after_agent 钩子调用）：差额退补 + 更新记录 + 流水。

        usage 需含 input_tokens / output_tokens / cached_input_tokens。
        actual_model_id 为实际使用模型（降级场景），None 表示未降级按原模型计费。
        request_id/provider_id/error_code/latency_ms 为成本归因字段（§3.1）。
        adjustment=True 表示对既有计费记录做补记（如推荐问题 token 计入同一条回复），
        仅执行差额退补与记录更新，跳过新增 consume 流水与异常检测，避免流水翻倍与误判。
        """
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
        cached_tokens = usage.get("cached_input_tokens", 0)

        # 降级计费：按实际模型换算（价格来源 sys_ai_model_price，见 AI模型管理 §2.12）
        settle_model = actual_model_id or model_id
        calc = await rate_provider.calculate(
            db, settle_model, provider_id, input_tokens, output_tokens, cached_tokens
        )
        actual_credits = calc["credits"]
        credits_saved = calc["credits_saved"]

        # 关联预扣的 chat 计费记录（pre_charge 创建）；缺失则新建
        billing = await self._find_chat_billing(
            db, user_id, conversation_id, message_id, bill_type
        )
        if billing is None:
            billing = await self.ai_billing_repository.create_billing(
                db,
                user_id=user_id,
                conversation_id=conversation_id,
                message_id=message_id,
                model=settle_model,
                bill_type=bill_type,
                request_id=request_id,
                provider_id=provider_id,
                error_code=error_code,
                latency_ms=latency_ms,
            )
            estimated = 0
        else:
            estimated = billing.pre_deduct or 0

        # 差额退补：difference = 预估 - 实际
        difference = estimated - actual_credits
        if difference > 0:
            # 多扣：退还配额与余额
            await quota_service.refund(user_id, difference)
            await balance_service.refund(db, user_id, difference)
        elif difference < 0:
            # 少扣：额外扣减（余额不足扣至 0 并标记欠费）
            extra = -difference
            await quota_service.deduct(user_id, extra)
            await balance_service.deduct(db, user_id, extra)

        # 更新计费记录（含成本归因字段，只写非空值）
        update_data: dict = {
            "credits": actual_credits,
            "quota_consumed": actual_credits,
            "credits_saved": credits_saved,
            "model": settle_model,
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_tokens,
            "output_tokens": output_tokens,
        }
        if actual_model_id:
            update_data["actual_model"] = model_id
        if request_id:
            update_data["request_id"] = request_id
        if provider_id:
            update_data["provider_id"] = provider_id
        if error_code:
            update_data["error_code"] = error_code
        if latency_ms is not None:
            update_data["latency_ms"] = latency_ms
        await self.ai_billing_repository.update(db, billing, update_data)

        # 补记路径（adjustment=True）：仅更新既有记录，不新增 consume 流水、不做异常检测
        if not adjustment:
            # 写入积分流水（source=consume，balance_after 取结算后余额）
            balance = await balance_service.get_balance(db, user_id)
            await self.ai_credit_log_repository.create_log(
                db,
                user_id=user_id,
                source="consume",
                amount=Decimal(-actual_credits),
                balance_after=balance,
                related_id=billing.id,
                reason=f"AI 对话消耗（{settle_model}）",
            )

            # 完整异常检测（单次超高/突发峰值/空回复高耗，后端实现 §4.7；
            # 内部尽力而为，失败不阻断结算主流程）
            daily_limit, monthly_limit = await quota_service.get_limits(db, user_id)
            await billing_anomaly_service.check(
                db,
                user_id,
                billing,
                monthly_limit=monthly_limit,
                daily_limit=daily_limit,
            )

            # 成本核算回填（成本线异步，失败不阻断结算主链路；db 为 None 的直连测试场景跳过）
            try:
                if db is not None and settings.AI_BILLING_COST_CALC_ENABLE:
                    await cost_service.backfill_cost(db, billing.id)
            except Exception:
                logger.warning("成本核算回填失败 billing_id=%s", billing.id)

            # 发布对话完成事件：AI 使用激励成长值（一次对话一次，fire-and-forget）
            _publish_chat_completed(user_id)

        return {
            "billing_id": billing.id,
            "credits": actual_credits,
            "credits_saved": credits_saved,
            "quota_consumed": actual_credits,
            "model": settle_model,
            "actual_model": model_id if actual_model_id else None,
        }

    async def _find_chat_billing(self, 
        db,
        user_id: int,
        conversation_id: int,
        message_id: int,
        bill_type: str,
    ):
        """查找该消息对应 bill_type 的计费记录（优先 pre_charge 创建的预扣记录）"""
        if not message_id:
            return None
        records = await self.ai_billing_repository.list_by_message(db, message_id)
        for r in records:
            if r.bill_type == bill_type and r.user_id == user_id:
                return r
        return None



billing_service = BillingService()
