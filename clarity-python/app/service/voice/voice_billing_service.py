"""语音交互计费服务：ASR 按秒、TTS 按万字符计入 AI 积分（统一模型价格表）

与 AI 对话计费（billing_service 预扣-结算两阶段）不同，语音用量在调用完成前
不可知（边说边计秒、边合成边计字符），采用"调用前预估校验 + 调用后实扣"，
计费执行复用 unit_billing（与图像计费共用）。
"""

import math
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.ai_billing_repository import ai_billing_repository
from app.repository.ai_credit_log_repository import ai_credit_log_repository
from app.service.ai_model_price_service import ai_model_price_service
from app.service.billing.balance_service import balance_service
from app.service.billing.quota_service import quota_service
from app.service.billing.unit_billing import UnitBillingService

# TTS 计费按「万字符」计量：调用方传字符数，这里换算为计量单位
CHARS_PER_BILLING_UNIT = 10_000


class VoiceBillingService(UnitBillingService):
    """语音能力计费：ASR 按秒、TTS 按字符（经统一模型价格表）"""

    def __init__(
        self,
        ai_billing_repository=ai_billing_repository,
        ai_credit_log_repository=ai_credit_log_repository,
        balance_service=balance_service,
        quota_service=quota_service,
        price_service=None,
    ):
        super().__init__(
            ai_billing_repository=ai_billing_repository,
            ai_credit_log_repository=ai_credit_log_repository,
            balance_service=balance_service,
            quota_service=quota_service,
            price_service=price_service if price_service is not None else ai_model_price_service,
        )

    async def estimate_asr_credits(
        self, db: AsyncSession, model_id: str, provider_id: int, audio_seconds: float
    ) -> int:
        """ASR 预估积分：秒数向上取整后按单价计（保守预估，实扣按 half-up 常小于等于此值）"""
        return await self.estimate_credits(
            db, model_id, provider_id, Decimal(math.ceil(audio_seconds))
        )

    async def estimate_tts_credits(
        self, db: AsyncSession, model_id: str, provider_id: int, text_chars: int
    ) -> int:
        """TTS 预估积分：字符数换算万字符后向上取整计（保守预估）"""
        units = Decimal(text_chars).scaleb(-4).to_integral_value(rounding="ROUND_CEILING")
        return await self.estimate_credits(db, model_id, provider_id, units)

    async def charge_asr(
        self, db: AsyncSession, user_id: int, model_id: str, provider_id: int, audio_seconds: float
    ) -> int:
        """ASR 按音频时长（秒）实扣，返回消耗积分数"""
        return await self._charge(
            db,
            user_id,
            bill_type="asr",
            model_id=model_id,
            provider_id=provider_id,
            quantity=Decimal(str(audio_seconds)),
            quantity_unit="second",
            raw_quantity=audio_seconds,
            reason=f"语音识别消耗（{model_id}）",
        )

    async def charge_tts(
        self, db: AsyncSession, user_id: int, model_id: str, provider_id: int, text_chars: int
    ) -> int:
        """TTS 按合成文本字符数实扣，返回消耗积分数"""
        return await self._charge(
            db,
            user_id,
            bill_type="tts",
            model_id=model_id,
            provider_id=provider_id,
            quantity=Decimal(text_chars).scaleb(-4),
            quantity_unit="ten_thousand_chars",
            raw_quantity=text_chars,
            reason=f"语音合成消耗（{model_id}）",
        )


voice_billing_service = VoiceBillingService()
