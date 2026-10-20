"""子 Agent 实报实销计费（billing_service.settle_subagent）与统计聚合口径

口径：主图预扣-结算链路不变；子 Agent run 不预扣，完成后按实际用量扣减
配额/余额并创建 bill_type=chat_subagent 独立记录（归属主会话用户/消息，
与主 chat 记录经 bill_type 区分、不被 _find_chat_billing 复用）；
统计侧 chat_subagent 与 chat 同口径聚合。
"""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.ai_billing_repository import AiBillingRepository
from app.repository.ai_credit_log_repository import AiCreditLogRepository
from app.service.billing import billing_service as m

pytestmark = pytest.mark.requires_db

# 测试替身：仓储层已 mock，db 仅传参占位
_DB: AsyncSession = AsyncMock(spec=AsyncSession)


class _BillingRepoStub(AiBillingRepository):
    """测试替身：仅实现 create_billing（委托注入实现）。"""

    def __init__(self, create_billing):
        self._create_billing = create_billing

    async def create_billing(self, *args, **kwargs):
        return await self._create_billing(*args, **kwargs)


class _BillingRepoListStub(AiBillingRepository):
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


def _install_subagent_stubs(monkeypatch, captured, rate=None):
    captured.setdefault("create", None)
    captured.setdefault("log_kwargs", [])
    captured.setdefault("quota_deduct", [])
    captured.setdefault("balance_deduct", [])

    async def _create_billing(db, **kwargs):
        captured["create"] = kwargs
        return SimpleNamespace(id=9)

    async def _calculate(db, model, provider_id, it, ot, ct):
        captured["calc_model"] = model
        return rate if rate is not None else {"credits": 30, "credits_saved": 5}

    async def _deduct_quota(uid, credits):
        captured["quota_deduct"].append((uid, credits))

    async def _deduct_balance(db, uid, credits):
        captured["balance_deduct"].append((uid, credits))

    async def _get_balance(db, uid):
        return 900

    async def _create_log(db, **kwargs):
        captured["log_kwargs"].append(kwargs)

    svc = m.BillingService(
        ai_billing_repository=_BillingRepoStub(_create_billing),
        ai_credit_log_repository=_CreditLogRepoStub(_create_log),
    )
    monkeypatch.setattr(m, "rate_provider", SimpleNamespace(calculate=_calculate))
    monkeypatch.setattr(m, "quota_service", SimpleNamespace(deduct=_deduct_quota))
    monkeypatch.setattr(
        m,
        "balance_service",
        SimpleNamespace(deduct=_deduct_balance, get_balance=_get_balance),
    )
    return svc


class TestSettleSubagent:
    async def test_creates_independent_record_and_deducts_actual(self, monkeypatch):
        """实报实销：按实际用量扣减配额与余额，独立 chat_subagent 记录（无预扣）"""
        captured = {}
        svc = _install_subagent_stubs(monkeypatch, captured)

        result = await svc.settle_subagent(
            _DB,
            10,
            1,
            2,
            "sub-model",
            None,
            {"input_tokens": 100, "output_tokens": 50, "cached_input_tokens": 20},
        )

        record = captured["create"]
        assert record["bill_type"] == "chat_subagent"
        assert record["user_id"] == 10
        assert record["conversation_id"] == 1
        assert record["message_id"] == 2
        assert record["model"] == "sub-model"
        assert record["credits"] == 30
        assert record["credits_saved"] == 5
        assert record["quota_consumed"] == 30
        assert record["pre_deduct"] == 0
        assert record["input_tokens"] == 100
        assert record["cached_input_tokens"] == 20
        assert record["output_tokens"] == 50
        assert captured["quota_deduct"] == [(10, 30)]
        assert captured["balance_deduct"] == [(10, 30)]
        # 消费流水照常记录（财务口径与主图/语音实扣一致）
        assert captured["log_kwargs"][0]["amount"] == Decimal(-30)
        assert captured["log_kwargs"][0]["source"] == "consume"
        assert result["billing_id"] == 9

    async def test_degraded_uses_actual_model_for_metering(self, monkeypatch):
        captured = {}
        svc = _install_subagent_stubs(monkeypatch, captured)
        await svc.settle_subagent(_DB, 10, 1, 2, "sub-model", "fallback-model", {})
        assert captured["calc_model"] == "fallback-model"
        assert captured["create"]["model"] == "fallback-model"
        assert captured["create"]["actual_model"] == "sub-model"

    async def test_zero_credits_no_deduct_no_log(self, monkeypatch):
        """0 消耗（无用量/免费模型）：不扣减、不写流水，仍落归因记录"""
        captured = {}
        svc = _install_subagent_stubs(
            monkeypatch, captured, rate={"credits": 0, "credits_saved": 0}
        )
        await svc.settle_subagent(_DB, 10, 1, 2, "sub-model", None, {})
        assert captured["quota_deduct"] == []
        assert captured["balance_deduct"] == []
        assert captured["log_kwargs"] == []
        assert captured["create"]["credits"] == 0


class TestMainSettleIgnoresSubagentRecord:
    async def test_find_chat_billing_never_picks_subagent_record(self, monkeypatch):
        """主图结算只关联预扣的 chat 记录：同消息的 chat_subagent 记录不参与差额退补"""
        chat_record = SimpleNamespace(id=1, pre_deduct=100, bill_type="chat", user_id=10)
        sub_record = SimpleNamespace(id=2, pre_deduct=0, bill_type="chat_subagent", user_id=10)

        async def _list_by_message(db, message_id):
            return [chat_record, sub_record]

        captured = {"updated": None, "logs": []}

        async def _update(db, entity, data):
            captured["updated"] = entity.id

        async def _create_log(db, **kwargs):
            captured["logs"].append(kwargs)

        async def _calculate(db, model, provider_id, it, ot, ct):
            return {"credits": 60, "credits_saved": 0}

        async def _refund_quota(uid, diff):
            captured["quota_refund"] = diff

        async def _refund_balance(db, uid, diff):
            pass

        async def _get_limits(db, uid):
            return (0, 0)

        async def _get_balance(db, uid):
            return 900

        async def _check_anomaly(db, uid, record, monthly_limit=0, daily_limit=0):
            pass

        svc = m.BillingService(
            ai_billing_repository=_BillingRepoListStub(_list_by_message, _update),
            ai_credit_log_repository=_CreditLogRepoStub(_create_log),
        )
        monkeypatch.setattr(m, "rate_provider", SimpleNamespace(calculate=_calculate))
        monkeypatch.setattr(
            m, "quota_service", SimpleNamespace(refund=_refund_quota, get_limits=_get_limits)
        )
        monkeypatch.setattr(
            m, "balance_service", SimpleNamespace(refund=_refund_balance, get_balance=_get_balance)
        )
        monkeypatch.setattr(m, "billing_anomaly_service", SimpleNamespace(check=_check_anomaly))
        monkeypatch.setattr(m, "_publish_chat_completed", lambda uid: None)
        # db 非 None 时结算走成本回填分支：桩掉成本服务（本用例不验证成本核算）
        monkeypatch.setattr(m, "cost_service", SimpleNamespace(backfill_cost=AsyncMock()))

        await svc.settle(_DB, 10, 1, 2, "m1", None, {"input_tokens": 10, "output_tokens": 5})

        # 差额基于主图 chat 记录的预扣（100-60=40 退回），子记录不被结算覆盖
        assert captured["updated"] == 1
        assert captured["quota_refund"] == 40


class TestStatsAggregateSubagent:
    async def test_summary_includes_subagent_usage(self, db):
        """用户 summary 聚合主+子（子 Agent 消耗实际扣减了用户配额与余额）"""
        from app.models.entity.sys_ai_billing import SysAiBilling
        from app.service.billing.billing_stat_service import BillingStatService

        def _billing(user_id, bill_type, credits, it=100, ot=50):
            return SysAiBilling(
                user_id=user_id,
                model="m1",
                bill_type=bill_type,
                input_tokens=it,
                output_tokens=ot,
                credits=credits,
                quota_consumed=credits,
                pre_deduct=credits,
            )

        db.add_all(
            [
                _billing(1, "chat", 100),
                _billing(1, "chat_subagent", 50),
                _billing(1, "asr", 10),  # 语音计量口径不同，不入 token 类汇总
            ]
        )
        await db.flush()

        result = await BillingStatService().summary(db, 1, "day")
        assert result.total_credits == 150
        assert result.input_tokens == 200
        assert result.output_tokens == 100

    async def test_admin_stats_tokens_include_subagent(self, db):
        """管理员统计 token 口径含子 Agent 记录（缓存命中率等 chat 口径同样适用）"""
        from app.models.entity.sys_ai_billing import SysAiBilling
        from app.models.schema.ai_billing import BillingStatQuery
        from app.service.billing.billing_stat_service import BillingStatService

        db.add_all(
            [
                SysAiBilling(
                    user_id=1,
                    model="m1",
                    bill_type="chat",
                    input_tokens=1000,
                    cached_input_tokens=200,
                    output_tokens=500,
                    credits=10,
                ),
                SysAiBilling(
                    user_id=1,
                    model="m1",
                    bill_type="chat_subagent",
                    input_tokens=400,
                    cached_input_tokens=100,
                    output_tokens=200,
                    credits=5,
                ),
            ]
        )
        await db.flush()

        rows = await BillingStatService().stats(db, BillingStatQuery(group_by="billType"))
        by_type = {r.dimension: r for r in rows}
        chat = by_type["chat"]
        assert chat.total_input_tokens == 1000
        assert chat.total_output_tokens == 500
        assert chat.cache_hit_rate == pytest.approx(0.2)
        sub = by_type["chat_subagent"]
        assert sub.total_input_tokens == 400
        assert sub.total_output_tokens == 200
        assert sub.cache_hit_rate == pytest.approx(0.25)
