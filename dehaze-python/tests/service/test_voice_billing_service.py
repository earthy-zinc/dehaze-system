"""语音计费服务单元测试（计费金额不变量）。

语音为"调用前预估校验 + 调用后实扣"后扣费模式（无预扣退款环节）：
- ASR 按音频秒数计费：credits = ceil(seconds × VOICE_ASR_CREDITS_PER_SECOND)
- TTS 按合成字符数计费：credits = ceil(chars × VOICE_TTS_CREDITS_PER_CHAR)
- 实扣同时扣减配额与余额、落计费记录（quantity 记入 input_tokens 作用量字段）
  与积分流水；credits=0 时不扣减、不记流水
"""

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.entity.sys_ai_billing import SysAiBilling
from app.models.entity.sys_ai_credit_log import SysAiCreditLog
from app.repository.ai_billing_repository import AiBillingRepository
from app.repository.ai_credit_log_repository import AiCreditLogRepository
from app.service.billing.balance_service import BalanceService
from app.service.billing.quota_service import QuotaService
from app.service.voice.voice_billing_service import VoiceBillingService


def _db() -> AsyncSession:
    """无绑定真实会话：调用链上的配额/余额/仓储均为桩，db 仅作形参透传。"""
    return AsyncSession()


class StubQuotaService(QuotaService):
    def __init__(self):
        self.deducted: list[int] = []

    async def check_quota(self, db: AsyncSession, user_id: int, estimated_credits: int) -> bool:
        return True

    async def deduct(self, user_id: int, credits: int) -> None:
        self.deducted.append(credits)


class StubBalanceService(BalanceService):
    def __init__(self, balance=Decimal(10000)):
        self._balance = balance
        self.deducted: list[int] = []

    async def is_arrears(self, user_id: int) -> bool:
        return False

    async def check_balance(self, db: AsyncSession, user_id: int, estimated_credits: int) -> bool:
        return self._balance >= estimated_credits

    async def deduct(self, db: AsyncSession, user_id: int, credits: int) -> None:
        self.deducted.append(credits)
        self._balance -= credits

    async def get_balance(self, db: AsyncSession, user_id: int) -> Decimal:
        return self._balance


class StubBillingRepo(AiBillingRepository):
    def __init__(self):
        self.created: list[dict] = []
        self.last: SysAiBilling | None = None

    async def create_billing(self, db: AsyncSession, **kwargs) -> SysAiBilling:
        self.created.append(kwargs)
        self.last = SysAiBilling(id=9001 + len(self.created))
        return self.last


class StubCreditLogRepo(AiCreditLogRepository):
    def __init__(self):
        self.logs: list[dict] = []

    async def create_log(self, db: AsyncSession, **kwargs) -> SysAiCreditLog:
        self.logs.append(kwargs)
        return SysAiCreditLog(id=7001 + len(self.logs))


def _make_svc():
    quota, balance = StubQuotaService(), StubBalanceService()
    billing_repo, credit_repo = StubBillingRepo(), StubCreditLogRepo()
    svc = VoiceBillingService(
        ai_billing_repository=billing_repo,
        ai_credit_log_repository=credit_repo,
        balance_service=balance,
        quota_service=quota,
    )
    return svc, quota, balance, billing_repo, credit_repo


@pytest.mark.parametrize("seconds", [1, 10, 0.5, 3.2, 61])
@pytest.mark.asyncio
async def test_charge_asr_credits_invariant(seconds):
    """计费不变量（固定费率）：ASR credits = ceil(秒数 × VOICE_ASR_CREDITS_PER_SECOND)，
    quantity = ceil(秒数) 记入 input_tokens 作用量字段"""
    import math

    svc, quota, balance, billing_repo, _ = _make_svc()

    credits = await svc.charge_asr(_db(), 1, seconds)

    assert credits == math.ceil(seconds * settings.VOICE_ASR_CREDITS_PER_SECOND)
    record = billing_repo.created[-1]
    assert record["bill_type"] == "asr"
    assert record["credits"] == credits
    assert record["input_tokens"] == math.ceil(seconds)
    assert quota.deducted == [credits]
    assert balance.deducted == [credits]


@pytest.mark.parametrize("chars", [1, 100, 9999, 3])
@pytest.mark.asyncio
async def test_charge_tts_credits_invariant(chars):
    """计费不变量（固定费率）：TTS credits = ceil(字符数 × VOICE_TTS_CREDITS_PER_CHAR)"""
    import math

    svc, quota, _balance, billing_repo, credit_repo = _make_svc()

    credits = await svc.charge_tts(_db(), 1, chars)

    assert credits == math.ceil(chars * settings.VOICE_TTS_CREDITS_PER_CHAR)
    record = billing_repo.created[-1]
    assert record["bill_type"] == "tts"
    assert record["input_tokens"] == chars
    assert quota.deducted == [credits]
    # 积分流水金额与扣费一致（负数），并关联计费记录
    assert credit_repo.logs[-1]["amount"] == Decimal(-credits)
    assert billing_repo.last is not None
    assert credit_repo.logs[-1]["related_id"] == billing_repo.last.id


@pytest.mark.asyncio
async def test_charge_zero_credits_skips_deduction_and_log():
    """用量为 0 → credits=0：不扣配额/余额、不记积分流水，仅落计费记录"""
    svc, quota, balance, billing_repo, credit_repo = _make_svc()

    credits = await svc.charge_tts(_db(), 1, 0)

    assert credits == 0
    assert quota.deducted == []
    assert balance.deducted == []
    assert credit_repo.logs == []
    assert len(billing_repo.created) == 1


@pytest.mark.asyncio
async def test_ensure_balance_rejects_arrears_user():
    """欠费用户：调用前预校验直接拒绝（不进入配额/余额检查）"""

    class ArrearsBalance(StubBalanceService):
        async def is_arrears(self, user_id: int) -> bool:
            return True

    from app.core.code import ResultCode
    from app.core.exceptions import BusinessException

    svc, _, _, _, _ = _make_svc()
    svc.balance_service = ArrearsBalance()

    with pytest.raises(BusinessException) as exc:
        await svc.ensure_balance(_db(), 1, 10)

    assert exc.value.code == ResultCode.QUOTA_INSUFFICIENT
