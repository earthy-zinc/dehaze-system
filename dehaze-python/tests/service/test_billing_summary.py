"""用户端消耗汇总 summary 与计费明细 refundStatus 单元测试"""

import pytest

pytestmark = pytest.mark.requires_db

from app.models.entity.sys_ai_billing import SysAiBilling
from app.models.entity.sys_ai_refund import SysAiRefund
from app.models.schema.ai_billing import BillingRecordQuery
from app.repository.ai_billing_repository import ai_billing_repository
from app.repository.ai_refund_repository import ai_refund_repository
from app.service.billing.billing_record_service import BillingRecordService
from app.service.billing.billing_stat_service import BillingStatService


def _billing(user_id, *, model="gpt-4o", credits=100, input_tokens=1000, output_tokens=500,
             cached=0, saved=0, bill_type="chat"):
    return SysAiBilling(
        user_id=user_id,
        model=model,
        bill_type=bill_type,
        input_tokens=input_tokens,
        cached_input_tokens=cached,
        output_tokens=output_tokens,
        credits=credits,
        credits_saved=saved,
        quota_consumed=credits,
        pre_deduct=credits,
    )


class TestSummary:
    async def test_day_summary_aggregates_only_self(self, db):
        stat_svc = BillingStatService(ai_billing_repository=ai_billing_repository)
        db.add_all([
            _billing(1, credits=100, input_tokens=1000, output_tokens=500),
            _billing(1, credits=200, input_tokens=2000, output_tokens=1000,
                     model="gpt-4o-mini", saved=30, cached=200),
            _billing(2, credits=999),  # 其他用户不计入
        ])
        await db.flush()

        result = await stat_svc.summary(db, 1, "day")
        assert result.total_credits == 300
        assert result.input_tokens == 3000
        assert result.output_tokens == 1500
        assert len(result.trend) == 1
        assert result.trend[0].credits == 300
        assert len(result.model_distribution) == 2
        top = result.model_distribution[0]
        assert top.model == "gpt-4o-mini"  # 按消耗降序
        assert result.savings.cached_input_tokens == 200
        assert result.savings.credits_saved == 30

    async def test_month_summary_dimension(self, db):
        stat_svc = BillingStatService(ai_billing_repository=ai_billing_repository)
        db.add(_billing(1, credits=50))
        await db.flush()

        result = await stat_svc.summary(db, 1, "month")
        assert result.total_credits == 50
        assert len(result.trend) == 1

    async def test_invalid_dimension_rejected(self, db):
        stat_svc = BillingStatService(ai_billing_repository=ai_billing_repository)
        from app.core.code import ResultCode
        from app.core.exceptions import BusinessException

        with pytest.raises(BusinessException) as exc:
            await stat_svc.summary(db, 1, "year")
        assert exc.value.code == ResultCode.PARAM_ERROR


class TestRefundStatus:
    async def test_record_without_refund_has_zero_status(self, db):
        record_svc = BillingRecordService(
            ai_billing_repository=ai_billing_repository,
            ai_refund_repository=ai_refund_repository,
        )
        db.add(_billing(1))
        await db.flush()

        result = await record_svc.list_by_user(db, 1, BillingRecordQuery(page=1, size=10))
        assert result.list[0].refund_status == 0

    async def test_record_refund_status_pending_and_approved(self, db):
        record_svc = BillingRecordService(
            ai_billing_repository=ai_billing_repository,
            ai_refund_repository=ai_refund_repository,
        )
        pending = _billing(1, credits=100)
        approved = _billing(1, credits=200, model="gpt-4o-mini")
        db.add_all([pending, approved])
        await db.flush()
        db.add_all([
            SysAiRefund(user_id=1, billing_id=pending.id, amount=50, reason="误扣", status=1),
            SysAiRefund(user_id=1, billing_id=approved.id, amount=100, reason="误扣", status=2, auditor_id=1),
        ])
        await db.flush()

        result = await record_svc.list_by_user(db, 1, BillingRecordQuery(page=1, size=10))
        status_map = {r.id: r.refund_status for r in result.list}
        assert status_map[pending.id] == 1
        assert status_map[approved.id] == 2

    async def test_latest_refund_status_overrides(self, db):
        record_svc = BillingRecordService(
            ai_billing_repository=ai_billing_repository,
            ai_refund_repository=ai_refund_repository,
        )
        billing = _billing(1)
        db.add(billing)
        await db.flush()
        db.add_all([
            SysAiRefund(user_id=1, billing_id=billing.id, amount=50, reason="误扣", status=1),
            SysAiRefund(user_id=1, billing_id=billing.id, amount=50, reason="误扣", status=3, auditor_id=1),
        ])
        await db.flush()

        result = await record_svc.list_by_user(db, 1, BillingRecordQuery(page=1, size=10))
        assert result.list[0].refund_status == 3
