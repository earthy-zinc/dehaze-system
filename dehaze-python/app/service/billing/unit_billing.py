"""非 token 计量能力的积分计费共享实现（设计 §2.6：ASR 按秒 / TTS 按万字符 / 图像按张）。

语音与图像同为「调用前预估校验 + 调用成功后实扣」的后扣费模式：用量在调用完成前
不可知，故不做预扣；未配置价格计 0 分、不阻断调用，可观测性由三件套承载
（warning 日志 + 指标 `dehaze_voice_unpriced_total` + 对账特征
`bill_type IN ('asr','tts') AND quantity>0 AND credits=0`）——**不在计费行写配置类标记**：
`sys_ai_billing.error_code` 的语义是"调用失败错误码（成功为 NULL）"。
单价来自 `sys_ai_model_price`（非 token 单行 unit 档，随价格版本管理）。
"""

import logging
from datetime import datetime
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.metrics.billing_metrics import record_unpriced
from app.repository.ai_billing_repository import ai_billing_repository
from app.repository.ai_credit_log_repository import ai_credit_log_repository
from app.service.ai_model_price_service import ai_model_price_service
from app.service.billing.balance_service import balance_service
from app.service.billing.quota_service import quota_service

logger = logging.getLogger(__name__)


class UnitBillingService:
    """非 token 计量计费基类：预估 / 余额预校验 / 实扣（子类补充具体计量换算）"""

    def __init__(
        self,
        ai_billing_repository=ai_billing_repository,
        ai_credit_log_repository=ai_credit_log_repository,
        balance_service=balance_service,
        quota_service=quota_service,
        price_service=None,
    ):
        self.ai_billing_repository = ai_billing_repository
        self.ai_credit_log_repository = ai_credit_log_repository
        self.balance_service = balance_service
        self.quota_service = quota_service
        # 子类在自身模块解析默认价格服务（便于测试按模块桩替换）
        self.price_service = price_service if price_service is not None else ai_model_price_service

    async def estimate_credits(
        self, db: AsyncSession, model_id: str, provider_id: int, quantity: Decimal
    ) -> int:
        """按模型单价预估积分（未配置价格返回 0，用于调用前预校验）"""
        result = await self.price_service.calculate_unit_credits(
            db, model_id, provider_id, datetime.now(), quantity
        )
        return result["credits"] if result["configured"] else 0

    async def ensure_balance(self, db: AsyncSession, user_id: int, estimated_credits: int) -> None:
        """调用前余额预校验：欠费 / AI 积分限额 / 余额任一不满足即抛业务异常

        estimated_credits 为本次调用的预估积分（按上限时长/文本长度/张数估算）。
        """
        if await self.balance_service.is_arrears(user_id):
            raise BusinessException(ResultCode.QUOTA_INSUFFICIENT, "账户欠费，请充值后继续使用")
        if not await self.quota_service.check_quota(db, user_id, estimated_credits):
            raise BusinessException(
                ResultCode.QUOTA_INSUFFICIENT, "今日或本月 AI 积分配额不足，请升级会员或明日再试"
            )
        if not await self.balance_service.check_balance(db, user_id, estimated_credits):
            raise BusinessException(ResultCode.QUOTA_INSUFFICIENT, "积分余额不足，请充值后继续使用")

    async def _charge(
        self,
        db: AsyncSession,
        user_id: int,
        *,
        bill_type: str,
        model_id: str,
        provider_id: int,
        quantity: Decimal,
        quantity_unit: str,
        raw_quantity: float | int,
        reason: str,
    ) -> int:
        """实扣配额与余额并记录计费行与积分流水；未配置价格计 0 分并计指标（不阻断调用）"""
        result = await self.price_service.calculate_unit_credits(
            db, model_id, provider_id, datetime.now(), quantity
        )
        configured = result["configured"]
        credits = result["credits"] if configured else 0
        if not configured:
            # 未配置价格的可观测三件套：warning 日志（带定位字段）+ 指标 + 对账特征
            logger.warning(
                "模型未配置价格，本次不计费: model_id=%s provider_id=%s bill_type=%s "
                "billing_unit=%s raw_quantity=%s",
                model_id,
                provider_id,
                bill_type,
                result.get("billing_unit") or quantity_unit,
                raw_quantity,
            )
            record_unpriced(bill_type)

        # 预校验已按单价拒绝过欠费调用；此处到账后余额可能不足，允许扣减为负欠费
        # （与对话实扣口径一致）
        if credits > 0:
            await self.quota_service.deduct(user_id, credits)
            await self.balance_service.deduct(db, user_id, credits)

        billing = await self.ai_billing_repository.create_billing(
            db,
            user_id=user_id,
            model=model_id,
            bill_type=bill_type,
            quantity=quantity,
            quantity_unit=quantity_unit,
            credits=credits,
            quota_consumed=credits,
            pre_deduct=0,
            # 未配置价格是配置状态、不是调用失败：success 行 error_code 显式留 NULL，
            # 不得写 no_price 之类标记（可观测性见上方三件套）
            error_code=None,
        )

        if credits > 0:
            balance = await self.balance_service.get_balance(db, user_id)
            await self.ai_credit_log_repository.create_log(
                db,
                user_id=user_id,
                source="consume",
                amount=Decimal(-credits),
                balance_after=balance,
                related_id=billing.id,
                reason=reason,
            )

        logger.info(
            "计费完成: user_id=%s bill_type=%s model=%s quantity=%s%s credits=%s",
            user_id,
            bill_type,
            model_id,
            raw_quantity,
            "" if configured else "（未配置价格，按 0 分落账）",
            credits,
        )
        return credits
