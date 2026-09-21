from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.ai_billing_repository import AiBillingRepository
from app.repository.ai_credit_log_repository import AiCreditLogRepository
from app.service.billing import billing_service as m

_BILLING = SimpleNamespace(id=1, pre_deduct=100, bill_type="chat", user_id=1)


class _BillingRepoStub(AiBillingRepository):
    """测试替身：仅实现 list_by_message / update（委托注入实现）。"""

    def __init__(self, list_by_message, update):
        self._list_by_message = list_by_message
        self._update = update

    async def list_by_message(self, *args, **kwargs):
        return await self._list_by_message(*args, **kwargs)

    async def update(self, *args, **kwargs):
        return await self._update(*args, **kwargs)


class _CreditLogRepoStub(AiCreditLogRepository):
    """测试替身：仅实现 create_log（委托注入实现）。"""

    def __init__(self, create_log):
        self._create_log = create_log

    async def create_log(self, *args, **kwargs):
        return await self._create_log(*args, **kwargs)


def _install_settle_stubs(monkeypatch, captured, rate=None):
    captured.setdefault("update", None)
    captured.setdefault("log_kwargs", [])
    captured.setdefault("anomaly", 0)
    captured.setdefault("calc_model", None)

    async def _list_by_message(db, message_id):
        return [_BILLING]

    async def _update(db, entity, data):
        captured["update"] = data

    async def _create_log(db, **kwargs):
        captured["log_kwargs"].append(kwargs)

    async def _calculate(db, model, provider_id, it, ot, ct):
        captured["calc_model"] = model
        return rate if rate is not None else {"credits": 50, "credits_saved": 10}

    async def _refund_quota(uid, diff):
        return None

    async def _deduct_quota(uid, extra):
        return None

    async def _get_limits(db, uid):
        return (0, 0)

    async def _refund_balance(db, uid, diff):
        return None

    async def _deduct_balance(db, uid, extra):
        return None

    async def _get_balance(db, uid):
        return 900

    async def _check_anomaly(db, uid, record, monthly_limit=0, daily_limit=0):
        captured["anomaly"] += 1

    svc = m.BillingService(
        ai_billing_repository=_BillingRepoStub(_list_by_message, _update),
        ai_credit_log_repository=_CreditLogRepoStub(_create_log),
    )
    # 服务引用仍为方法体内模块级查找，故 patch 模块对象 m
    monkeypatch.setattr(m, "rate_provider", SimpleNamespace(calculate=_calculate))
    monkeypatch.setattr(
        m,
        "quota_service",
        SimpleNamespace(refund=_refund_quota, deduct=_deduct_quota, get_limits=_get_limits),
    )
    monkeypatch.setattr(
        m,
        "balance_service",
        SimpleNamespace(refund=_refund_balance, deduct=_deduct_balance, get_balance=_get_balance),
    )
    monkeypatch.setattr(m, "billing_anomaly_service", SimpleNamespace(check=_check_anomaly))

    # 成本核算回填与本组用例（归因字段/事件接线）无关，置为 no-op 保证结算路径确定性
    async def _backfill_cost(db, billing_id):
        return None

    monkeypatch.setattr(m.cost_service, "backfill_cost", _backfill_cost)
    # 对话完成事件默认拦截（接线测试单独覆盖验证发布调用），避免后台任务污染
    monkeypatch.setattr(m, "_publish_chat_completed", lambda uid: None)
    return svc


class TestSettleAttribution:
    async def test_settle_writes_attribution_fields(self, monkeypatch):
        captured = {}
        svc = _install_settle_stubs(monkeypatch, captured)

        await svc.settle(
            AsyncMock(spec=AsyncSession),
            user_id=1,
            conversation_id=2,
            message_id=3,
            model_id="gpt-4o",
            actual_model_id=None,
            usage={
                "input_tokens": 100,
                "output_tokens": 50,
                "cached_input_tokens": 20,
            },
            request_id="req-001",
            provider_id=7,
            error_code=None,
            latency_ms=1234,
        )

        data = captured["update"]
        assert data["request_id"] == "req-001"
        assert data["provider_id"] == 7
        assert data["latency_ms"] == 1234
        assert "error_code" not in data
        assert captured["log_kwargs"][0]["source"] == "consume"
        assert captured["log_kwargs"][0]["amount"] == Decimal(-50)
        assert captured["anomaly"] == 1

    async def test_settle_omits_none_attribution(self, monkeypatch):
        captured = {}
        svc = _install_settle_stubs(monkeypatch, captured)

        await svc.settle(
            AsyncMock(spec=AsyncSession),
            user_id=1,
            conversation_id=2,
            message_id=3,
            model_id="gpt-4o",
            actual_model_id=None,
            usage={},
        )

        data = captured["update"]
        for key in ("request_id", "provider_id", "error_code", "latency_ms"):
            assert key not in data

    async def test_settle_degraded_writes_actual_model(self, monkeypatch):
        captured = {}
        svc = _install_settle_stubs(monkeypatch, captured)

        await svc.settle(
            AsyncMock(spec=AsyncSession),
            user_id=1,
            conversation_id=2,
            message_id=3,
            model_id="gpt-4o",
            actual_model_id="claude-3-5-haiku",
            usage={"input_tokens": 100, "output_tokens": 50},
        )

        data = captured["update"]
        assert data["model"] == "claude-3-5-haiku"
        assert data["actual_model"] == "gpt-4o"
        assert captured["calc_model"] == "claude-3-5-haiku"

    async def test_settle_adjustment_skips_log_and_anomaly(self, monkeypatch):
        captured = {}
        svc = _install_settle_stubs(monkeypatch, captured, rate={"credits": 60, "credits_saved": 5})

        await svc.settle(
            AsyncMock(spec=AsyncSession),
            user_id=1,
            conversation_id=2,
            message_id=3,
            model_id="gpt-4o",
            actual_model_id=None,
            usage={"input_tokens": 120, "output_tokens": 60},
            adjustment=True,
        )

        assert captured["log_kwargs"] == []
        assert captured["anomaly"] == 0
        assert captured["update"]["credits"] == 60


class TestSettleChatCompletedEvent:
    """ai.chat.completed 事件发布接线：对话完成结算发布一次，补记不发布"""

    async def test_settle_publishes_chat_completed_once(self, monkeypatch):
        captured = {}
        svc = _install_settle_stubs(monkeypatch, captured)
        published: list[int] = []
        monkeypatch.setattr(m, "_publish_chat_completed", lambda uid: published.append(uid))

        await svc.settle(
            AsyncMock(spec=AsyncSession),
            user_id=1,
            conversation_id=2,
            message_id=3,
            model_id="gpt-4o",
            actual_model_id=None,
            usage={"input_tokens": 10, "output_tokens": 5},
        )

        assert published == [1]

    async def test_settle_adjustment_skips_publish(self, monkeypatch):
        captured = {}
        svc = _install_settle_stubs(monkeypatch, captured, rate={"credits": 60, "credits_saved": 5})
        published: list[int] = []
        monkeypatch.setattr(m, "_publish_chat_completed", lambda uid: published.append(uid))

        await svc.settle(
            AsyncMock(spec=AsyncSession),
            user_id=1,
            conversation_id=2,
            message_id=3,
            model_id="gpt-4o",
            actual_model_id=None,
            usage={"input_tokens": 120, "output_tokens": 60},
            adjustment=True,
        )

        assert published == []
