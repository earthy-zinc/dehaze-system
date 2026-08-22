import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.service.billing import balance_service as bm
from app.service.billing import bill_service as blm
from app.service.billing import billing_anomaly_service as am
from app.service.billing import billing_service as bs
from app.service.billing import estimate_service as es
from app.service.billing import quota_service as qm
from app.service.billing import rate_provider as rpm
from app.service.billing import refund_service as rf
from app.service.billing.quota_service import _quota_keys_and_ttl
from tests.stubs import StubAsyncSession, async_ret, fake_redis

_REDIS_MODULES = (bm, qm, es, rpm, am, blm)


def _bind_redis(monkeypatch, redis, modules=_REDIS_MODULES):
    async def _get():
        return redis

    for mod in modules:
        monkeypatch.setattr(mod, "get_redis_client", _get)


async def _bind_fake_redis(monkeypatch, modules, data=None):
    redis = await fake_redis(data)
    _bind_redis(monkeypatch, redis, modules)
    return redis


def _model(rates=None, status=1, max_output=4096):
    r = rates or (1.0, 4.0, 0.5)
    return SimpleNamespace(
        input_rate=r[0], output_rate=r[1], cached_rate=r[2],
        max_output_tokens=max_output, status=status,
    )


def _member_benefit(daily=10000, monthly=100000, status=1):
    return SimpleNamespace(
        status=status, ai_credits_daily=daily, ai_credits_monthly=monthly
    )


def _patch_limits(monkeypatch, benefit, member="exists"):
    monkeypatch.setattr(
        qm.member_repository, "get_by_user_id", async_ret(
            SimpleNamespace(level_code="level_0") if member else None
        )
    )
    monkeypatch.setattr(
        qm.member_benefit_repository, "get_by_level_code", async_ret(benefit)
    )


async def _quota_setup(monkeypatch, benefit=None, member="exists", data=None):
    redis = await _bind_fake_redis(monkeypatch, (qm,), data)
    _patch_limits(monkeypatch, _member_benefit() if benefit is None else benefit, member=member)
    return redis


class TestRateCalculate:
    @staticmethod
    async def _bind_rate(monkeypatch, model, cache=None):
        redis = await _bind_fake_redis(monkeypatch, (rpm,), cache)
        monkeypatch.setattr(rpm.ai_model_repository, "get_by_model_id", async_ret(model))
        return redis

    async def test_basic_formula_and_credits_saved(self, monkeypatch):
        await self._bind_rate(monkeypatch, _model())
        calc = await rpm.RateProvider.calculate(None, "gpt-4o", 1000, 500, 200)
        assert calc["credits"] == int(1000 * 1.0 + 200 * 0.5 + 500 * 4.0)
        assert calc["credits_saved"] == int(200 * (1.0 - 0.5))

    async def test_model_not_found_safe_zero(self, monkeypatch):
        await self._bind_rate(monkeypatch, None)
        calc = await rpm.RateProvider.calculate(None, "unknown", 1000, 500, 200)
        assert calc["credits"] == 0 and calc["credits_saved"] == 0

    async def test_rate_cache_hit(self, monkeypatch):
        await self._bind_rate(
            monkeypatch,
            _model(),
            cache={
                "ai:rate:gpt-4o": '{"input_rate": 2.0, "output_rate": 8.0, '
                '"cached_rate": 1.0, "max_output_tokens": 1024}'
            },
        )
        called = {"db": 0}

        async def _no_call(db, m):
            called["db"] += 1
            return _model()

        monkeypatch.setattr(rpm.ai_model_repository, "get_by_model_id", _no_call)
        rates = await rpm.RateProvider.get_rates(None, "gpt-4o")
        assert rates["input_rate"] == 2.0
        assert called["db"] == 0


class TestQuotaService:
    async def test_check_quota_daily_exceeded(self, monkeypatch):
        redis = await _quota_setup(monkeypatch)
        daily_key, _, _, _ = _quota_keys_and_ttl(1)
        await redis.set(daily_key, "9000")
        assert not await qm.quota_service.check_quota(None, 1, 3000)

    async def test_check_quota_monthly_exceeded(self, monkeypatch):
        redis = await _quota_setup(monkeypatch)
        _, monthly_key, _, _ = _quota_keys_and_ttl(1)
        await redis.set(monthly_key, "98000")
        assert not await qm.quota_service.check_quota(None, 1, 3000)

    async def test_check_quota_sufficient(self, monkeypatch):
        await _quota_setup(monkeypatch)
        assert await qm.quota_service.check_quota(None, 1, 3000)

    async def test_check_quota_zero_limit_means_unlimited(self, monkeypatch):
        redis = await _quota_setup(monkeypatch, member=None)
        daily_key, monthly_key, _, _ = _quota_keys_and_ttl(1)
        await redis.set(daily_key, "999999999")
        await redis.set(monthly_key, "999999999")
        assert await qm.quota_service.check_quota(None, 1, 3000) is True

    async def test_get_limits_no_member(self, monkeypatch):
        _patch_limits(monkeypatch, _member_benefit(), member=None)
        assert await qm.quota_service.get_limits(None, 1) == (0, 0)

    async def test_get_limits_disabled_benefit(self, monkeypatch):
        _patch_limits(monkeypatch, _member_benefit(status=0))
        assert await qm.quota_service.get_limits(None, 1) == (0, 0)

    async def test_pre_deduct_insufficient(self, monkeypatch):
        redis = await _quota_setup(monkeypatch)
        daily_key, monthly_key, _, _ = _quota_keys_and_ttl(1)
        await redis.set(daily_key, "9000")
        assert not await qm.quota_service.pre_deduct(None, 1, 2000)
        assert int(await redis.get(daily_key)) == 9000
        assert await redis.get(monthly_key) is None

    async def test_pre_deduct_sufficient(self, monkeypatch):
        redis = await _quota_setup(monkeypatch)
        daily_key, monthly_key, _, _ = _quota_keys_and_ttl(1)
        assert await qm.quota_service.pre_deduct(None, 1, 2000)
        assert int(await redis.get(daily_key)) == 2000
        assert int(await redis.get(monthly_key)) == 2000


class TestBalanceService:
    async def test_pre_deduct_insufficient_rollback(self, monkeypatch):
        redis = await _bind_fake_redis(monkeypatch, (bm,), {"ai:balance:1": "100"})
        ok = await bm.balance_service.pre_deduct(None, 1, 300)
        assert not ok
        assert int(await redis.get("ai:balance:1")) == 100

    async def test_deduct_overdraw_marks_arrears(self, monkeypatch):
        redis = await _bind_fake_redis(monkeypatch, (bm,), {"ai:balance:1": "100"})
        cas = {"amount": None}

        async def _deduct_cas(db, uid, amount):
            cas["amount"] = amount

        monkeypatch.setattr(bm.balance_service, "_deduct_cas", _deduct_cas)
        from app.repository.ai_credit_log_repository import ai_credit_log_repository as log_repo

        monkeypatch.setattr(log_repo, "create_log", async_ret(None))
        await bm.balance_service.deduct(None, 1, 300)
        assert int(await redis.get("ai:balance:1")) == 0
        assert int(await redis.get("ai:arrears:1")) == 1
        assert cas["amount"] == 100

    async def test_increase_clears_arrears(self, monkeypatch):
        redis = await _bind_fake_redis(
            monkeypatch, (bm,), {"ai:balance:1": "0", "ai:arrears:1": "1"}
        )
        logs = []

        async def _create_log(db, **kwargs):
            logs.append(kwargs)

        monkeypatch.setattr(bm.balance_service, "_increase_cas", async_ret(None))
        from app.repository.ai_credit_log_repository import ai_credit_log_repository as log_repo

        monkeypatch.setattr(log_repo, "create_log", _create_log)
        await bm.balance_service.increase(None, 1, 500, source="recharge")
        assert await redis.get("ai:arrears:1") is None
        assert logs and logs[0]["source"] == "recharge" and logs[0]["amount"] == 500

    async def test_is_arrears(self, monkeypatch):
        await _bind_fake_redis(monkeypatch, (bm,), {"ai:arrears:1": "1"})
        assert await bm.balance_service.is_arrears(1)
        assert not await bm.balance_service.is_arrears(2)


class TestPreCharge:
    @staticmethod
    def _patch_base(monkeypatch, estimated=1000):
        monkeypatch.setattr(
            bs.estimate_service, "estimate_credits", async_ret(estimated)
        )
        monkeypatch.setattr(
            bs.ai_billing_repository, "create_billing", async_ret(SimpleNamespace(id=99))
        )
        monkeypatch.setattr(bs.balance_service, "is_arrears", async_ret(False))
        monkeypatch.setattr(bs.quota_service, "check_quota", async_ret(True))
        monkeypatch.setattr(bs.balance_service, "check_balance", async_ret(True))
        monkeypatch.setattr(bs.quota_service, "pre_deduct", async_ret(True))
        monkeypatch.setattr(bs.balance_service, "pre_deduct", async_ret(True))

    async def test_arrears_blocks(self, monkeypatch):
        self._patch_base(monkeypatch)
        monkeypatch.setattr(bs.balance_service, "is_arrears", async_ret(True))
        result = await bs.billing_service.pre_charge(None, 1, 2, 3, "hi", "gpt-4o")
        assert result["stop_reason"] == "arrears"

    async def test_quota_fail_records_anomaly(self, monkeypatch):
        self._patch_base(monkeypatch)
        monkeypatch.setattr(bs.quota_service, "check_quota", async_ret(False))
        fail_calls = []

        async def _fail(uid):
            fail_calls.append(uid)

        monkeypatch.setattr(bs.billing_anomaly_service, "record_quota_fail", _fail)
        result = await bs.billing_service.pre_charge(None, 1, 2, 3, "hi", "gpt-4o")
        assert result["stop_reason"] == "quota_exceeded"
        assert fail_calls == [1]

    async def test_balance_insufficient_blocks(self, monkeypatch):
        self._patch_base(monkeypatch)
        monkeypatch.setattr(bs.balance_service, "check_balance", async_ret(False))
        result = await bs.billing_service.pre_charge(None, 1, 2, 3, "hi", "gpt-4o")
        assert result["stop_reason"] == "balance_exceeded"

    async def test_balance_prededuct_fail_rolls_back_quota(self, monkeypatch):
        self._patch_base(monkeypatch)
        monkeypatch.setattr(bs.balance_service, "pre_deduct", async_ret(False))
        refunds = []

        async def _refund(uid, c):
            refunds.append(c)

        monkeypatch.setattr(bs.quota_service, "refund", _refund)
        result = await bs.billing_service.pre_charge(None, 1, 2, 3, "hi", "gpt-4o")
        assert result["stop_reason"] == "balance_exceeded"
        assert refunds == [1000]

    async def test_success_returns_context(self, monkeypatch):
        self._patch_base(monkeypatch, estimated=2000)
        result = await bs.billing_service.pre_charge(None, 1, 2, 3, "hi", "gpt-4o")
        assert result["billing_id"] == 99
        assert result["budget_pool"] == 2000
        assert result["remaining_budget"] == 2000


class TestSettleDiff:
    @staticmethod
    def _patch_settle(monkeypatch, actual_credits, billing=None):
        billing = billing or SimpleNamespace(
            id=1, pre_deduct=2000, bill_type="chat", user_id=1
        )
        ops = {"quota_refund": 0, "balance_refund": 0, "quota_deduct": 0, "balance_deduct": 0}

        class _Repo:
            async def list_by_message(self, db, mid):
                return [billing]

            async def update(self, db, entity, data):
                for k, v in data.items():
                    setattr(entity, k, v)
                return entity

        class _Quota:
            async def refund(self, uid, c):
                ops["quota_refund"] += c

            async def deduct(self, uid, c):
                ops["quota_deduct"] += c

            async def get_limits(self, db, uid):
                return (0, 0)

        class _Balance:
            async def refund(self, db, uid, c):
                ops["balance_refund"] += c

            async def deduct(self, db, uid, c):
                ops["balance_deduct"] += c

            async def get_balance(self, db, uid):
                return 900

        class _Anomaly:
            async def check(self, uid, record, monthly_limit=0, daily_limit=0):
                return None

        monkeypatch.setattr(bs, "ai_billing_repository", _Repo())
        monkeypatch.setattr(
            bs, "ai_credit_log_repository", SimpleNamespace(create_log=async_ret(None))
        )
        monkeypatch.setattr(
            bs, "RateProvider",
            SimpleNamespace(calculate=async_ret(
                {"credits": actual_credits, "credits_saved": 0}
            )),
        )
        monkeypatch.setattr(bs, "quota_service", _Quota())
        monkeypatch.setattr(bs, "balance_service", _Balance())
        monkeypatch.setattr(bs, "billing_anomaly_service", _Anomaly())
        return billing, ops

    async def test_overestimate_refunds_difference(self, monkeypatch):
        _, ops = self._patch_settle(monkeypatch, actual_credits=1500)
        result = await bs.billing_service.settle(
            None, 1, 2, 3, "gpt-4o", None, {"input_tokens": 100, "output_tokens": 50}
        )
        assert result["quota_consumed"] == 1500
        assert ops["quota_refund"] == 500 and ops["balance_refund"] == 500
        assert ops["quota_deduct"] == 0 and ops["balance_deduct"] == 0

    async def test_underestimate_deducts_extra(self, monkeypatch):
        billing = SimpleNamespace(id=1, pre_deduct=1000, bill_type="chat", user_id=1)
        _, ops = self._patch_settle(monkeypatch, actual_credits=2000, billing=billing)
        await bs.billing_service.settle(
            None, 1, 2, 3, "gpt-4o", None, {"input_tokens": 100, "output_tokens": 50}
        )
        assert ops["quota_deduct"] == 1000 and ops["balance_deduct"] == 1000
        assert ops["quota_refund"] == 0 and ops["balance_refund"] == 0

    async def test_zero_usage_refunds_all(self, monkeypatch):
        _, ops = self._patch_settle(monkeypatch, actual_credits=0)
        result = await bs.billing_service.settle(None, 1, 2, 3, "gpt-4o", None, {})
        assert result["credits"] == 0
        assert ops["quota_refund"] == 2000 and ops["balance_refund"] == 2000


class TestAnomalyRules:
    @staticmethod
    def _record(credits=0, input_tokens=0, output_tokens=0):
        return SimpleNamespace(
            credits=credits, input_tokens=input_tokens, output_tokens=output_tokens
        )

    async def test_single_high_triggers(self, monkeypatch):
        redis = await _bind_fake_redis(monkeypatch, (am,))
        await am.billing_anomaly_service.check(
            1, self._record(credits=15000), monthly_limit=100000
        )
        assert int(await redis.get("ai:anomaly:count:single_high:1")) == 1

    async def test_burst_peak_triggers(self, monkeypatch):
        redis = await _bind_fake_redis(monkeypatch, (am,))
        await am.billing_anomaly_service.check(
            1, self._record(credits=6000), daily_limit=10000
        )
        assert int(await redis.get("ai:anomaly:count:burst_peak:1")) == 1

    async def test_empty_reply_high_cost_triggers(self, monkeypatch):
        redis = await _bind_fake_redis(monkeypatch, (am,))
        await am.billing_anomaly_service.check(
            1, self._record(input_tokens=15000, output_tokens=0)
        )
        assert int(await redis.get("ai:anomaly:count:empty_reply_high_cost:1")) == 1

    async def test_normal_usage_no_alert(self, monkeypatch):
        redis = await _bind_fake_redis(monkeypatch, (am,))
        await am.billing_anomaly_service.check(
            1, self._record(credits=500, input_tokens=1000, output_tokens=200),
            monthly_limit=100000, daily_limit=10000,
        )
        keys = [k async for k in redis.scan_iter(match="ai:anomaly:count:*")]
        assert not keys

    async def test_consecutive_quota_fail_alert(self, monkeypatch):
        redis = await _bind_fake_redis(monkeypatch, (am,))
        for _ in range(10):
            await am.billing_anomaly_service.record_quota_fail(1)
        assert int(await redis.get("ai:anomaly:count:consecutive_quota_fail:1")) == 1

    async def test_redis_unavailable_safe(self, monkeypatch):
        async def _broken():
            raise ConnectionError("redis down")

        monkeypatch.setattr(am, "get_redis_client", _broken)
        await am.billing_anomaly_service.check(
            1, self._record(credits=999999), monthly_limit=1
        )
        await am.billing_anomaly_service.record_quota_fail(1)


class TestRefundService:
    async def test_apply_refund_zero_or_negative_amount_rejected(self):
        from app.core.code import ResultCode
        from app.core.exceptions import BusinessException

        with pytest.raises(BusinessException) as exc:
            await rf.refund_service.apply_refund(None, 1, 5, 0, "误扣")
        assert exc.value.code == ResultCode.PARAM_ERROR
        with pytest.raises(BusinessException) as exc2:
            await rf.refund_service.apply_refund(None, 1, 5, -100, "误扣")
        assert exc2.value.code == ResultCode.PARAM_ERROR

    async def test_apply_refund_duplicate_rejected(self, monkeypatch):
        from app.core.code import ResultCode
        from app.core.exceptions import BusinessException

        monkeypatch.setattr(
            rf.ai_billing_repository, "get_by_id",
            async_ret(SimpleNamespace(id=5, user_id=1)),
        )
        monkeypatch.setattr(
            rf.ai_refund_repository, "get_pending_by_billing_id", async_ret(object())
        )
        with pytest.raises(BusinessException) as exc:
            await rf.refund_service.apply_refund(None, 1, 5, 100, "误扣")
        assert exc.value.code == ResultCode.AI_REFUND_ALREADY_EXISTS

    async def test_apply_refund_other_users_record_rejected(self, monkeypatch):
        from app.core.exceptions import BusinessException

        monkeypatch.setattr(
            rf.ai_billing_repository, "get_by_id",
            async_ret(SimpleNamespace(id=5, user_id=2)),
        )
        monkeypatch.setattr(
            rf.ai_refund_repository, "get_pending_by_billing_id", async_ret(None)
        )
        with pytest.raises(BusinessException):
            await rf.refund_service.apply_refund(None, 1, 5, 100, "误扣")

    async def test_audit_refund_already_audited_rejected(self, monkeypatch):
        from app.core.code import ResultCode
        from app.core.exceptions import BusinessException

        monkeypatch.setattr(
            rf.ai_refund_repository, "get_by_id",
            async_ret(SimpleNamespace(id=9, status=2)),
        )
        with pytest.raises(BusinessException) as exc:
            await rf.refund_service.audit_refund(None, 9, True, None, 2)
        assert exc.value.code == ResultCode.REFUND_AUDIT_FAILED

    async def test_audit_refund_approve_increases_balance(self, monkeypatch):
        refund = SimpleNamespace(
            id=9, status=1, user_id=1, billing_id=5, amount=2000, reason="误扣",
            auditor_id=None, audit_remark=None,
        )
        monkeypatch.setattr(rf.ai_refund_repository, "get_by_id", async_ret(refund))
        increases = []

        async def _increase(db, uid, amount, source=None, related_id=None,
                            reason=None, operator_id=None):
            increases.append(
                {"uid": uid, "amount": amount, "source": source, "related": related_id}
            )
            return 3000

        monkeypatch.setattr(rf.balance_service, "increase", _increase)

        result = await rf.refund_service.audit_refund(StubAsyncSession(), 9, True, "同意", 2)
        assert result.status == 2
        assert increases[0]["source"] == "refund"
        assert increases[0]["related"] == 5


class TestBillService:
    @staticmethod
    def _patch_bill(monkeypatch, by_type, by_source, balances):
        monkeypatch.setattr(
            blm.ai_billing_repository,
            "sum_credits_by_user_group_by_bill_type",
            async_ret(by_type),
        )
        monkeypatch.setattr(
            blm.ai_credit_log_repository,
            "sum_amount_by_user_and_source",
            async_ret(by_source),
        )
        monkeypatch.setattr(
            blm.ai_credit_log_repository,
            "get_balance_at_or_before",
            async_ret(balances),
        )

        class _Cache:
            def __init__(self, redis):
                self.redis = redis

            async def get_json(self, key, default=None):
                raw = await self.redis.get(key)
                return json.loads(raw) if raw else default

            async def set_json(self, key, value, ttl=0):
                await self.redis.set(key, json.dumps(value, default=str))
                return True

            async def delete(self, key):
                await self.redis.delete(key)
                return True

        monkeypatch.setattr(blm, "CacheService", _Cache)

    async def test_generate_monthly_bill_aggregates(self, monkeypatch):
        await _bind_fake_redis(monkeypatch, (blm,))
        self._patch_bill(
            monkeypatch,
            by_type=[
                {"bill_type": "chat", "credits": 3000},
                {"bill_type": "tool_llm", "credits": 500},
            ],
            by_source={
                "recharge": Decimal(1000),
                "vip_gift": Decimal(200),
                "vip_gift_expire": Decimal(-50),
                "refund": Decimal(300),
                "consume": Decimal(-3500),
            },
            balances=Decimal(1500),
        )
        bill = await blm.bill_service.generate_monthly_bill(None, 1, "2026-07")
        assert bill.total_consume == 3500
        assert bill.total_recharge == 1150
        assert bill.total_refund == 300
        assert bill.item_summary == {"chat": 3000, "tool_llm": 500}
        assert bill.balance_end == 1500

    async def test_get_bill_cache_hit(self, monkeypatch):
        redis = await _bind_fake_redis(monkeypatch, (blm,))
        self._patch_bill(monkeypatch, by_type=[], by_source={}, balances=Decimal(0))
        cached = {
            "user_id": 1, "month": "2026-07", "total_consume": 100,
            "total_recharge": 0, "total_refund": 0, "balance_start": "0",
            "balance_end": "0", "item_summary": {},
        }
        await redis.set("ai:bill:1:2026-07", json.dumps(cached, default=str))
        bill = await blm.bill_service.get_bill(None, 1, "2026-07")
        assert bill.total_consume == 100

    async def test_get_bill_empty_history_month_not_found(self, monkeypatch):
        from app.core.code import ResultCode
        from app.core.exceptions import BusinessException

        redis = await _bind_fake_redis(monkeypatch, (blm,))
        self._patch_bill(monkeypatch, by_type=[], by_source={}, balances=Decimal(0))
        with pytest.raises(BusinessException) as exc:
            await blm.bill_service.get_bill(None, 1, "2020-01")
        assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND
        assert await redis.get("ai:bill:1:2020-01") is None
        with pytest.raises(BusinessException) as exc2:
            await blm.bill_service.get_bill(None, 1, "2020-01")
        assert exc2.value.code == ResultCode.RESOURCE_NOT_FOUND

    async def test_get_bill_stale_empty_cache_not_found(self, monkeypatch):
        from app.core.code import ResultCode
        from app.core.exceptions import BusinessException

        redis = await _bind_fake_redis(monkeypatch, (blm,))
        self._patch_bill(monkeypatch, by_type=[], by_source={}, balances=Decimal(0))
        cached = {
            "user_id": 1, "month": "2020-01", "total_consume": 0,
            "total_recharge": 0, "total_refund": 0, "balance_start": "0",
            "balance_end": "0", "item_summary": {},
        }
        await redis.set("ai:bill:1:2020-01", json.dumps(cached, default=str))
        with pytest.raises(BusinessException) as exc:
            await blm.bill_service.get_bill(None, 1, "2020-01")
        assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND


class TestEstimateService:
    async def test_new_conversation_conservative_estimate(self, monkeypatch):
        await _bind_fake_redis(monkeypatch, (es,))
        rates = {
            "input_rate": 1.0, "output_rate": 4.0,
            "cached_rate": 0.5, "max_output_tokens": 4096,
        }
        monkeypatch.setattr(es.RateProvider, "get_rates", async_ret(rates))
        estimated = await es.estimate_service.estimate_credits(
            None, 1, 100, "请帮我总结今天的工作会议记录并提取待办事项", "gpt-4o"
        )
        expected = int(706 * 1.0 + 1228.8 * 4.0)
        assert estimated == expected

    async def test_ctx_avg_read_failure_treated_as_no_history(self, monkeypatch):
        from redis.exceptions import ConnectionError as RedisConnectionError

        class _BrokenRedis:
            async def get(self, key):
                raise RedisConnectionError("redis read broken")

            async def set(self, key, value, ex=None):
                return True

        async def _get():
            return _BrokenRedis()

        monkeypatch.setattr(es, "get_redis_client", _get)
        rates = {
            "input_rate": 1.0, "output_rate": 4.0,
            "cached_rate": 0.5, "max_output_tokens": 4096,
        }
        monkeypatch.setattr(es.RateProvider, "get_rates", async_ret(rates))
        estimated = await es.estimate_service.estimate_credits(
            None, 1, 100, "hi", "gpt-4o"
        )
        assert estimated > 0


class TestBillingJobs:
    async def test_clear_vip_gift_expire_negative_amount(self, monkeypatch):
        logs = []

        async def _distinct(db, source, start, end):
            return [1]

        async def _sum(db, uid, start, end):
            return {"vip_gift": Decimal(300)}

        balance_seq = [Decimal(300), Decimal(100)]

        async def _balance(db, uid):
            return balance_seq.pop(0)

        async def _deduct(db, uid, amount):
            logs.append({"deduct": amount})

        async def _create_log(db, **kwargs):
            logs.append(kwargs)

        import app.service.billing.balance_service as bal_mod
        from app.repository.ai_credit_log_repository import ai_credit_log_repository as log_repo

        monkeypatch.setattr(log_repo, "distinct_user_ids_by_source", _distinct)
        monkeypatch.setattr(log_repo, "sum_amount_by_user_and_source", _sum)
        monkeypatch.setattr(log_repo, "create_log", _create_log)
        monkeypatch.setattr(bal_mod.balance_service, "get_balance", _balance)
        monkeypatch.setattr(bal_mod.balance_service, "deduct", _deduct)

        import app.infrastructure.job.handlers as h

        result = await h.clear_vip_gift_expire()
        assert "清零=1" in result
        create_log_call = [entry for entry in logs if "source" in entry][0]
        assert create_log_call["amount"] == -300
        assert create_log_call["source"] == "vip_gift_expire"

    async def test_grant_vip_monthly_gift_by_level(self, monkeypatch):
        benefits = [
            SimpleNamespace(level_code="level_1", status=1, vip_gift_credits=500),
            SimpleNamespace(level_code="level_0", status=1, vip_gift_credits=0),
            SimpleNamespace(level_code="level_2", status=0, vip_gift_credits=999),
        ]
        grants = []

        import app.repository.member_benefit_repository as ben_mod
        import app.repository.member_repository as mem_mod
        import app.service.billing.recharge_service as rec_mod

        async def _list_all(db):
            return benefits

        async def _list_members(db, level_code, offset=0, limit=500):
            if level_code == "level_1" and offset == 0:
                return [SimpleNamespace(user_id=11), SimpleNamespace(user_id=12)]
            return []

        async def _grant(db, user_id, amount):
            grants.append({"user_id": user_id, "amount": amount})

        monkeypatch.setattr(ben_mod.member_benefit_repository, "list_all", _list_all)
        monkeypatch.setattr(mem_mod.member_repository, "list_active_by_level", _list_members)
        monkeypatch.setattr(rec_mod.recharge_service, "grant_vip_monthly_gift", _grant)

        import app.infrastructure.job.handlers as h

        result = await h.grant_vip_monthly_gift()
        assert "发放用户数=2" in result
        assert grants == [
            {"user_id": 11, "amount": 500},
            {"user_id": 12, "amount": 500},
        ]


class TestRegisterTrialCredits:
    async def test_register_grants_trial_credits(self, monkeypatch):
        import app.repository.member_repository as mem_mod
        import app.service.auth_service as auth_mod
        import app.service.billing.recharge_service as rec_mod
        from app.config import settings
        from app.repository.role_repository import role_repository as role_repo
        from app.repository.user_repository import user_repository as user_repo

        grants = []

        async def _grant(db, user_id):
            grants.append(user_id)
            return 100

        async def _get_or_init(db, user_id):
            return SimpleNamespace(user_id=user_id)

        monkeypatch.setattr(rec_mod.recharge_service, "grant_trial_credits", _grant)
        monkeypatch.setattr(mem_mod.member_repository, "get_or_init_member", _get_or_init)
        monkeypatch.setattr(user_repo, "check_username_exists", async_ret(False))
        monkeypatch.setattr(
            role_repo,
            "get_enabled_by_code",
            async_ret(SimpleNamespace(id=7, data_scope=0)),
        )

        redis = await fake_redis({
            f"{settings.CAPTCHA_KEY_PREFIX}test-key": "abcd",
        })

        async def _hash(pw):
            return "hashed"

        monkeypatch.setattr(auth_mod, "hash_password_async", _hash)

        result = await auth_mod.auth_service.register(
            StubAsyncSession(), redis, "newuser", "Passw0rd!", "昵称", "test-key", "ABCD"
        )
        assert result["user"]["id"] == 1
        assert grants == [1]
