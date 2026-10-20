"""未配置价格的可观测三件套之二：指标 `dehaze_voice_unpriced_total{model_type}`。

不依赖 DB：价格换算与扣减/落账仓库全部以桩注入，只断言"未配置价格必增指标、配价后不增"。
（warning 日志与"计 0 分落账 + 不扣减不流水"由 test_voice_billing_service 覆盖。）
"""

from decimal import Decimal
from types import SimpleNamespace

import pytest
from prometheus_client import REGISTRY
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_billing import SysAiBilling
from app.models.entity.sys_ai_credit_log import SysAiCreditLog
from app.repository.ai_billing_repository import AiBillingRepository
from app.repository.ai_credit_log_repository import AiCreditLogRepository
from app.service.billing.balance_service import BalanceService
from app.service.billing.quota_service import QuotaService
from app.service.billing.unit_billing import UnitBillingService

METRIC_NAME = "dehaze_voice_unpriced_total"


class StubBillingRepo(AiBillingRepository):
    def __init__(self):
        self.created: list[dict] = []

    async def create_billing(self, db: AsyncSession, **kwargs) -> SysAiBilling:
        self.created.append(kwargs)
        return SysAiBilling(id=9001)


class StubCreditLogRepo(AiCreditLogRepository):
    def __init__(self):
        self.logs: list[dict] = []

    async def create_log(self, db: AsyncSession, **kwargs) -> SysAiCreditLog:
        self.logs.append(kwargs)
        return SysAiCreditLog(id=7001)


class StubQuotaService(QuotaService):
    def __init__(self):
        self.deducted: list[int] = []

    async def deduct(self, user_id: int, credits: int) -> None:
        self.deducted.append(credits)


class StubBalanceService(BalanceService):
    def __init__(self):
        self.deducted: list[int] = []

    async def deduct(self, db: AsyncSession, user_id: int, credits: int) -> None:
        self.deducted.append(credits)

    async def get_balance(self, db: AsyncSession, user_id: int) -> Decimal:
        return Decimal(1000)


class StubPriceService:
    def __init__(self, configured: bool):
        self.configured = configured

    async def calculate_unit_credits(
        self, db: AsyncSession, model_id: str, provider_id: int, at_time, quantity: Decimal
    ) -> dict:
        return {
            "credits": 3 if self.configured else 0,
            "configured": self.configured,
            "billing_unit": "second",
        }


def _unpriced_count(model_type: str) -> float:
    return REGISTRY.get_sample_value(METRIC_NAME, {"model_type": model_type}) or 0.0


def _make_svc(configured: bool) -> tuple[UnitBillingService, SimpleNamespace]:
    stubs = SimpleNamespace(
        billing=StubBillingRepo(),
        credit_log=StubCreditLogRepo(),
        quota=StubQuotaService(),
        balance=StubBalanceService(),
    )
    svc = UnitBillingService(
        ai_billing_repository=stubs.billing,
        ai_credit_log_repository=stubs.credit_log,
        balance_service=stubs.balance,
        quota_service=stubs.quota,
        price_service=StubPriceService(configured),
    )
    return svc, stubs


async def _charge_asr(svc: UnitBillingService, bill_type: str = "asr") -> None:
    await svc._charge(
        AsyncSession(),
        1,
        bill_type=bill_type,
        model_id="sensevoice",
        provider_id=78,
        quantity=Decimal("5"),
        quantity_unit="second",
        raw_quantity=5,
        reason="语音识别消耗",
    )


@pytest.mark.asyncio
async def test_unconfigured_price_increments_metric_with_bill_type_label():
    """未配置价格：按计费类型（asr/tts）计数一次，供"配了价却仍是 0 分"告警"""
    svc, stubs = _make_svc(configured=False)
    asr_before, tts_before = _unpriced_count("asr"), _unpriced_count("tts")

    await _charge_asr(svc)
    await _charge_asr(svc, bill_type="tts")

    assert [row["credits"] for row in stubs.billing.created] == [0, 0]
    assert _unpriced_count("asr") == asr_before + 1
    assert _unpriced_count("tts") == tts_before + 1


@pytest.mark.asyncio
async def test_configured_price_does_not_increment_metric():
    """负向对照：配了价（含显式 0 价的免费模型）不得计入未定价指标"""
    svc, stubs = _make_svc(configured=True)
    before = _unpriced_count("asr")

    await _charge_asr(svc)

    assert stubs.billing.created[0]["credits"] == 3
    assert stubs.quota.deducted == [3]
    assert stubs.credit_log.logs
    assert _unpriced_count("asr") == before
