"""语音交互计费服务：ASR 按音频秒数、TTS 按合成字符数计入 AI 积分

与 AI 对话计费（billing_service 预扣-结算两阶段）不同，语音用量在调用完成前
不可知（边说边计秒、边合成边计字符），采用"调用前预估校验 + 调用后实扣"：
- ensure_balance：调用前校验欠费/配额/余额，不足抛业务异常拒绝调用
- charge_asr / charge_tts：调用完成后按实际用量扣减配额与余额并记录流水

计费单价暂以配置项过渡（需求规格 §8 待确认事项 5：最终由 AI 计费管理模块统一定价）。
"""

import logging
import math
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.repository.ai_billing_repository import ai_billing_repository
from app.repository.ai_credit_log_repository import ai_credit_log_repository
from app.service.billing.balance_service import balance_service
from app.service.billing.quota_service import quota_service

logger = logging.getLogger(__name__)


class VoiceBillingService:
    """语音能力计费：ASR 按秒、TTS 按字符"""

    def __init__(
        self,
        ai_billing_repository=ai_billing_repository,
        ai_credit_log_repository=ai_credit_log_repository,
        balance_service=balance_service,
        quota_service=quota_service,
    ):
        self.ai_billing_repository = ai_billing_repository
        self.ai_credit_log_repository = ai_credit_log_repository
        self.balance_service = balance_service
        self.quota_service = quota_service

    async def ensure_balance(self, db: AsyncSession, user_id: int, estimated_credits: int) -> None:
        """调用前余额预校验，欠费/配额/余额任一不满足即抛业务异常

        estimated_credits 为本次调用的预估积分（按上限时长/文本长度估算）。
        """
        if await self.balance_service.is_arrears(user_id):
            raise BusinessException(ResultCode.QUOTA_INSUFFICIENT, "账户欠费，请充值后继续使用")
        if not await self.quota_service.check_quota(db, user_id, estimated_credits):
            raise BusinessException(
                ResultCode.QUOTA_INSUFFICIENT, "今日或本月 AI 积分配额不足，请升级会员或明日再试"
            )
        if not await self.balance_service.check_balance(db, user_id, estimated_credits):
            raise BusinessException(ResultCode.QUOTA_INSUFFICIENT, "积分余额不足，请充值后继续使用")

    async def charge_asr(self, db: AsyncSession, user_id: int, audio_seconds: float) -> int:
        """ASR 按音频时长（秒）实扣，返回消耗积分数"""
        credits = math.ceil(audio_seconds * settings.VOICE_ASR_CREDITS_PER_SECOND)
        return await self._charge(
            db,
            user_id,
            bill_type="asr",
            model="funasr",
            quantity=int(math.ceil(audio_seconds)),
            credits=credits,
            reason="语音识别消耗（FunASR）",
        )

    async def charge_tts(self, db: AsyncSession, user_id: int, text_chars: int) -> int:
        """TTS 按合成文本字符数实扣，返回消耗积分数"""
        credits = math.ceil(text_chars * settings.VOICE_TTS_CREDITS_PER_CHAR)
        return await self._charge(
            db,
            user_id,
            bill_type="tts",
            model="piper",
            quantity=text_chars,
            credits=credits,
            reason="语音合成消耗（本地 Piper）",
        )

    async def _charge(self, 
        db: AsyncSession,
        user_id: int,
        *,
        bill_type: str,
        model: str,
        quantity: int,
        credits: int,
        reason: str,
    ) -> int:
        """实扣配额与余额并记录计费记录与积分流水（quantity 记入 input_tokens 作用量字段）"""
        credits = max(credits, 0)
        if credits > 0:
            await quota_service.deduct(user_id, credits)
            await balance_service.deduct(db, user_id, credits)

        billing = await ai_billing_repository.create_billing(
            db,
            user_id=user_id,
            model=model,
            bill_type=bill_type,
            input_tokens=quantity,
            credits=credits,
            quota_consumed=credits,
            pre_deduct=0,
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
            "语音计费完成: user_id=%s bill_type=%s quantity=%s credits=%s",
            user_id, bill_type, quantity, credits,
        )
        return credits


voice_billing_service = VoiceBillingService()
