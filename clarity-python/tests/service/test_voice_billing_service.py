"""语音计费服务单元测试（计费金额不变量，阶段 2 统一模型价格口径）。

语音为"调用前预估校验 + 调用后实扣"后扣费模式（无预扣退款环节）：
- ASR 按秒 / TTS 按万字符，积分经 AiModelPriceService.calculate_unit_credits 换算
  （单价来自 sys_ai_model_price 的 unit 档；换算口径由 test_ai_model_price 覆盖，
  此处以桩预设换算结果，考察编排不变量）
- 预估保守（数量向上取整），实扣按 half-up（常小于等于预估）
- 未配置价格不阻断：计 0 积分，计费行 error_code 保持 NULL（不写配置类标记）
- 实扣同时扣减配额与余额、落计费行（model=统一 model_id，quantity/quantity_unit
  记入专用列）与积分流水；credits=0 时不扣减、不记流水
"""

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_billing import SysAiBilling
from app.models.entity.sys_ai_credit_log import SysAiCreditLog
from app.repository.ai_billing_repository import AiBillingRepository
from app.repository.ai_credit_log_repository import AiCreditLogRepository
from app.service.billing import unit_billing as unit_billing_module
from app.service.billing.balance_service import BalanceService
from app.service.billing.quota_service import QuotaService
from app.service.voice import voice_billing_service as vbs_module
from app.service.voice.voice_billing_service import VoiceBillingService

pytestmark = pytest.mark.requires_db

USER_ID = 1
MODEL_ID = "sensevoice"
PROVIDER_ID = 78


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


def _stub_pricing(
    monkeypatch, *, credits: int, configured: bool = True, billing_unit: str = "second"
):
    """预设 calculate_unit_credits 换算结果（换算口径由 test_ai_model_price 独立覆盖）"""

    async def _calc(db, model_id, provider_id, at_time, quantity):
        return {"credits": credits, "configured": configured, "billing_unit": billing_unit}

    monkeypatch.setattr(
        vbs_module,
        "ai_model_price_service",
        type("P", (), {"calculate_unit_credits": staticmethod(_calc)})(),
    )


# ===== 实扣：换算结果落账不变量 =====


@pytest.mark.asyncio
async def test_charge_asr_records_model_and_quantity_columns(db, monkeypatch):
    """ASR 实扣：计费行记统一 model_id 与 quantity/quantity_unit 专用列（非 input_tokens）"""
    _stub_pricing(monkeypatch, credits=3)
    svc, quota, balance, billing_repo, credit_repo = _make_svc()

    credits = await svc.charge_asr(db, USER_ID, MODEL_ID, PROVIDER_ID, 2.4)

    assert credits == 3
    assert quota.deducted == [3]
    assert balance.deducted == [3]
    assert len(billing_repo.created) == 1
    row = billing_repo.created[0]
    assert row["model"] == MODEL_ID
    assert row["bill_type"] == "asr"
    assert row["quantity"] == Decimal("2.4")
    assert row["quantity_unit"] == "second"
    assert row["error_code"] is None
    assert row.get("input_tokens") is None  # 未复用 token 列存语音量
    assert credit_repo.logs
    assert credit_repo.logs[0]["amount"] == Decimal(-3)


@pytest.mark.asyncio
async def test_charge_tts_converts_chars_to_ten_thousand(db, monkeypatch):
    """TTS 实扣：字符数换算为万字符计量单位落 quantity"""
    _stub_pricing(monkeypatch, credits=100, billing_unit="ten_thousand_chars")
    svc, _, _, billing_repo, _ = _make_svc()

    await svc.charge_tts(db, USER_ID, "huayan", PROVIDER_ID, 250)

    row = billing_repo.created[0]
    assert row["quantity"] == Decimal("0.0250")  # 250 字符 = 0.025 万字符
    assert row["quantity_unit"] == "ten_thousand_chars"
    assert row["bill_type"] == "tts"


@pytest.mark.asyncio
async def test_charge_with_no_price_records_zero_credits_and_null_error_code(db, monkeypatch):
    """未配置价格：不阻断调用，计 0 积分、计费行 error_code 保持 NULL，无扣减与流水

    未定价是**配置状态**而非调用失败（`sys_ai_billing.error_code` 语义＝"调用失败错误码，
    成功为 NULL"），写 no_price 之类标记会把成功计费行标成失败、制造静默错账。
    未定价可观测性由三件套承载：warning 日志 + `dehaze_voice_unpriced_total` +
    对账特征 `bill_type IN ('asr','tts') AND quantity>0 AND credits=0`。
    """
    _stub_pricing(monkeypatch, credits=0, configured=False)
    svc, quota, balance, billing_repo, credit_repo = _make_svc()

    credits = await svc.charge_asr(db, USER_ID, MODEL_ID, PROVIDER_ID, 5)

    assert credits == 0
    assert quota.deducted == []
    assert balance.deducted == []
    row = billing_repo.created[0]
    assert row["credits"] == 0
    assert row["error_code"] is None  # 正向：成功计费行 error_code 为 NULL
    assert credit_repo.logs == []


def test_no_price_marker_is_removed_without_alias():
    """旧标记 NO_PRICE_ERROR_CODE='no_price'（语义污染）已删除，且不得以别名方式保留"""
    assert not hasattr(unit_billing_module, "NO_PRICE_ERROR_CODE")


# ===== 预估：保守向上取整 =====


@pytest.mark.asyncio
async def test_estimate_asr_credits_conservative_ceiling(db, monkeypatch):
    """ASR 预估：秒数向上取整后换算（2.4 秒按 3 计），实扣换算常小于等于预估"""
    seen: list[Decimal] = []

    async def _calc(db, model_id, provider_id, at_time, quantity):
        seen.append(quantity)
        return {"credits": int(quantity), "configured": True, "billing_unit": "second"}

    monkeypatch.setattr(
        vbs_module,
        "ai_model_price_service",
        type("P", (), {"calculate_unit_credits": staticmethod(_calc)})(),
    )
    svc, *_ = _make_svc()

    assert await svc.estimate_asr_credits(db, MODEL_ID, PROVIDER_ID, 2.4) == 3
    assert seen == [Decimal(3)]  # 数量先 ceil 再换算

    # 实扣按原始数量换算（half-up 由 calculate_unit_credits 承担）
    assert await svc.charge_asr(db, USER_ID, MODEL_ID, PROVIDER_ID, 2.4) == 2
    assert seen[-1] == Decimal("2.4")


@pytest.mark.asyncio
async def test_estimate_tts_credits_ceils_to_billing_unit(db, monkeypatch):
    """TTS 预估：字符数换算万字符后向上取整（1 字符按 1 万字符保守预估）"""
    seen: list[Decimal] = []

    async def _calc(db, model_id, provider_id, at_time, quantity):
        seen.append(quantity)
        return {
            "credits": int(quantity) * 100,
            "configured": True,
            "billing_unit": "ten_thousand_chars",
        }

    monkeypatch.setattr(
        vbs_module,
        "ai_model_price_service",
        type("P", (), {"calculate_unit_credits": staticmethod(_calc)})(),
    )
    svc, *_ = _make_svc()

    assert await svc.estimate_tts_credits(db, "huayan", PROVIDER_ID, 1) == 100
    assert seen == [Decimal(1)]


@pytest.mark.asyncio
async def test_estimate_with_no_price_returns_zero(db, monkeypatch):
    """未配置价格：预估为 0（ensure_balance 对 0 积分恒放行，不拦截调用）"""
    _stub_pricing(monkeypatch, credits=0, configured=False)
    svc, *_ = _make_svc()

    assert await svc.estimate_asr_credits(db, MODEL_ID, PROVIDER_ID, 10) == 0


# ===== 预校验：欠费/配额/余额任一不满足即拒绝 =====


@pytest.mark.asyncio
async def test_ensure_balance_rejects_arrears(db, monkeypatch):
    _stub_pricing(monkeypatch, credits=10)
    svc, _, balance, _, _ = _make_svc()

    async def _arrears(user_id):
        return True

    balance.is_arrears = _arrears
    from app.core.code import ResultCode
    from app.core.exceptions import BusinessException

    with pytest.raises(BusinessException) as exc_info:
        await svc.ensure_balance(db, USER_ID, 10)
    assert exc_info.value.code == ResultCode.QUOTA_INSUFFICIENT


@pytest.mark.asyncio
async def test_ensure_balance_rejects_insufficient_quota(db, monkeypatch):
    _stub_pricing(monkeypatch, credits=10)
    svc, _, _, _, _ = _make_svc()

    class _NoQuota(StubQuotaService):
        async def check_quota(self, db, user_id, estimated_credits):
            return False

    svc.quota_service = _NoQuota()
    from app.core.exceptions import BusinessException

    with pytest.raises(BusinessException):
        await svc.ensure_balance(db, USER_ID, 10)


@pytest.mark.asyncio
async def test_ensure_balance_rejects_insufficient_balance(db, monkeypatch):
    _stub_pricing(monkeypatch, credits=10)
    svc, _, balance, _, _ = _make_svc()
    balance._balance = Decimal(5)
    from app.core.exceptions import BusinessException

    with pytest.raises(BusinessException):
        await svc.ensure_balance(db, USER_ID, 10)


@pytest.mark.asyncio
async def test_charge_zero_credits_no_deduction_no_log(db, monkeypatch):
    """换算为 0 积分（如 0 秒音频）：落计费行但不扣减、不记流水"""
    _stub_pricing(monkeypatch, credits=0, configured=True)
    svc, quota, _, billing_repo, credit_repo = _make_svc()

    await svc.charge_asr(db, USER_ID, MODEL_ID, PROVIDER_ID, 0)

    assert quota.deducted == []
    assert billing_repo.created[0]["credits"] == 0
    assert credit_repo.logs == []
