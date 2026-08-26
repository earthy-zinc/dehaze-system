from decimal import Decimal
from types import SimpleNamespace

from app.service.billing import billing_service as m

_BILLING = SimpleNamespace(id=1, pre_deduct=100, bill_type="chat", user_id=1)


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

    repo = SimpleNamespace(list_by_message=_list_by_message, update=_update)
    svc = m.BillingService(
        ai_billing_repository=repo,
        ai_credit_log_repository=SimpleNamespace(create_log=_create_log),
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
    # 对话完成事件默认拦截（接线测试单独覆盖验证发布调用），避免后台任务污染
    monkeypatch.setattr(m, "_publish_chat_completed", lambda uid: None)
    return svc


class TestSettleAttribution:
    async def test_settle_writes_attribution_fields(self, monkeypatch):
        captured = {}
        svc = _install_settle_stubs(monkeypatch, captured)

        await svc.settle(
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
            user_id=1,
            conversation_id=2,
            message_id=3,
            model_id="gpt-4o",
            actual_model_id=None,
            usage={"input_tokens": 120, "output_tokens": 60},
            adjustment=True,
        )

        assert published == []
