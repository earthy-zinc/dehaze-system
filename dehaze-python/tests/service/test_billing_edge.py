"""计费边缘强化：并发扣减原子性、限额边界、差额回补、滚动预算、脏语料、跨月账单口径"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_billing import SysAiBilling
from app.models.entity.sys_ai_credit_log import SysAiCreditLog
from app.models.entity.sys_ai_refund import SysAiRefund
from app.models.entity.sys_member import SysMember
from app.models.entity.sys_member_benefit import SysMemberBenefit
from app.repository.ai_credit_log_repository import AiCreditLogRepository
from app.repository.member_benefit_repository import MemberBenefitRepository
from app.repository.member_repository import MemberRepository
from app.repository.user_repository import UserRepository
from app.service.billing import bill_service as blm
from app.service.billing import billing_service as bs
from app.service.billing import refund_service as rf
from app.service.billing.balance_service import BalanceService
from app.service.billing.quota_service import QuotaService, _quota_keys_and_ttl

pytestmark = pytest.mark.requires_db

# 直连测试的 db 替身：仓储被 mock、db 不真正落库，用真实会话实例占位
_DB = AsyncSession()


class _MemberRepo(MemberRepository):
    """测试替身：仅覆写 get_by_user_id（返回注入的会员记录）。"""

    def __init__(self, member=None):
        self._member = member

    async def get_by_user_id(self, db, user_id):
        return self._member


class _BenefitRepo(MemberBenefitRepository):
    """测试替身：仅覆写 get_by_level_code（返回注入的权益配置）。"""

    def __init__(self, benefit=None):
        self._benefit = benefit

    async def get_by_level_code(self, db, level_code):
        return self._benefit


class _UserRepo(UserRepository):
    """测试替身：仅覆写 get_credits_balance_and_version（返回注入余额）。"""

    def __init__(self, balance):
        self._balance = balance

    async def get_credits_balance_and_version(self, db, user_id):
        return (self._balance, 1)


class _CreditLogRepo(AiCreditLogRepository):
    """测试替身：仅覆写 create_log（返回内存流水）。"""

    async def create_log(
        self,
        db,
        *,
        user_id,
        source,
        amount,
        balance_after,
        related_id=None,
        reason=None,
        operator_id=None,
    ):
        return SysAiCreditLog(
            user_id=user_id,
            source=source,
            amount=amount,
            balance_after=balance_after,
            related_id=related_id,
            reason=reason,
            operator_id=operator_id,
        )


def _quota_svc(daily=10000, monthly=100000):
    return QuotaService(
        member_repository=_MemberRepo(SysMember(level_code="level_0")),
        member_benefit_repository=_BenefitRepo(
            SysMemberBenefit(status=1, ai_credits_daily=daily, ai_credits_monthly=monthly)
        ),
    )


def _balance_svc(balance=Decimal("1000")):
    return BalanceService(
        user_repository=_UserRepo(balance),
        ai_credit_log_repository=_CreditLogRepo(),
    )


class TestQuotaConcurrency:
    """T-AB-028 并发不超扣：Redis Lua 原子预扣"""

    async def test_concurrent_pre_deduct_never_overdraws(self, mock_redis):
        redis = mock_redis
        svc = _quota_svc(daily=5000, monthly=0)  # 月限额 0 = 不限量
        results = await asyncio.gather(*[svc.pre_deduct(_DB, 1, 3000) for _ in range(10)])
        daily_key, _, _, _ = _quota_keys_and_ttl(1)
        successes = sum(1 for r in results if r)
        # 剩余 5000 仅容得下一条 3000，其余必须被整体回滚
        assert successes == 1
        assert int(await redis.get(daily_key)) == 3000
        assert int(await redis.get(daily_key)) <= 5000

    async def test_concurrent_unit_deducts_exact_total(self, mock_redis):
        redis = mock_redis
        svc = _quota_svc(daily=1000, monthly=0)
        results = await asyncio.gather(*[svc.pre_deduct(_DB, 1, 1) for _ in range(1200)])
        daily_key, _, _, _ = _quota_keys_and_ttl(1)
        # 不变量：成功数恰等于限额，不超扣也不少放
        assert sum(1 for r in results if r) == 1000
        assert int(await redis.get(daily_key)) == 1000

    async def test_pre_deduct_monthly_exceeded_rolls_back_daily(self, mock_redis):
        """日侧充足但月侧超限：Lua 整体回滚，日"已用"不得残留增量"""
        redis = mock_redis
        svc = _quota_svc(daily=10000, monthly=100000)
        daily_key, monthly_key, _, _ = _quota_keys_and_ttl(1)
        await redis.set(monthly_key, "99000")
        assert not await svc.pre_deduct(_DB, 1, 3000)
        assert int(await redis.get(daily_key)) == 0
        assert int(await redis.get(monthly_key)) == 99000


class TestQuotaBoundary:
    """日/月限额恰好用尽与差 1 边界"""

    async def test_check_quota_exact_limit_passes(self, mock_redis):
        redis = mock_redis
        svc = _quota_svc(daily=10000, monthly=100000)
        daily_key, monthly_key, _, _ = _quota_keys_and_ttl(1)
        await redis.set(daily_key, "7000")
        await redis.set(monthly_key, "97000")
        assert await svc.check_quota(_DB, 1, 3000) is True

    async def test_check_quota_off_by_one_fails(self, mock_redis):
        redis = mock_redis
        svc = _quota_svc(daily=10000, monthly=100000)
        daily_key, monthly_key, _, _ = _quota_keys_and_ttl(1)
        await redis.set(daily_key, "7000")
        await redis.set(monthly_key, "97000")
        assert await svc.check_quota(_DB, 1, 3001) is False

    async def test_pre_deduct_fills_limit_then_rejects(self, mock_redis):
        redis = mock_redis
        svc = _quota_svc(daily=1000, monthly=1000)
        daily_key, _monthly_key, _, _ = _quota_keys_and_ttl(1)
        assert await svc.pre_deduct(_DB, 1, 1000) is True
        assert int(await redis.get(daily_key)) == 1000
        # 已用 == 限额后再扣 1 积分必须拒绝且无副作用
        assert await svc.pre_deduct(_DB, 1, 1) is False
        assert int(await redis.get(daily_key)) == 1000

    async def test_quota_refund_decrements_both_keys(self, mock_redis):
        redis = mock_redis
        svc = _quota_svc()
        daily_key, monthly_key, _, _ = _quota_keys_and_ttl(1)
        await redis.set(daily_key, "2000")
        await redis.set(monthly_key, "20000")
        await svc.refund(1, 500)
        assert int(await redis.get(daily_key)) == 1500
        assert int(await redis.get(monthly_key)) == 19500

    async def test_quota_deduct_initializes_missing_keys(self, mock_redis):
        redis = mock_redis
        svc = _quota_svc()
        daily_key, monthly_key, _, _ = _quota_keys_and_ttl(1)
        await svc.deduct(1, 500)
        assert int(await redis.get(daily_key)) == 500
        assert int(await redis.get(monthly_key)) == 500


class TestBalanceConcurrency:
    """余额并发预扣不超扣、防负余额、坏缓存自愈"""

    async def test_concurrent_pre_deduct_no_overdraw(self, mock_redis):
        redis = mock_redis
        svc = _balance_svc(Decimal("1000"))
        results = await asyncio.gather(*[svc.pre_deduct(_DB, 1, 300) for _ in range(10)])
        successes = sum(1 for r in results if r)
        # 不变量：成功数×300 ≤ 1000，余额不出现负数
        assert successes == 3
        assert int(await redis.get("ai:balance:1")) == 1000 - successes * 300

    async def test_pre_deduct_recovers_from_non_integer_cache(self, mock_redis):
        """历史坏值（如 "100.00"）触发 ResponseError 后清缓存由 MySQL 整数化回填重试"""
        redis = mock_redis
        await redis.set("ai:balance:1", "100.00")
        svc = _balance_svc(Decimal("100"))
        assert await svc.pre_deduct(_DB, 1, 50) is True
        assert int(await redis.get("ai:balance:1")) == 50

    async def test_pre_deduct_insufficient_leaves_no_side_effect(self, mock_redis):
        redis = mock_redis
        svc = _balance_svc(Decimal("100"))
        assert await svc.pre_deduct(_DB, 1, 300) is False
        assert int(await redis.get("ai:balance:1")) == 100


class TestCheckBudget:
    """滚动预算校验（before_model 钩子）：第 3 步超出剩余预算触发中止"""

    async def test_insufficient_budget_interrupts_with_quota_type(self):
        result = await bs.billing_service.check_budget(
            {"billing_context": {"remaining_budget": 1000}}, 1500
        )
        assert result is not None
        assert result["stop_reason"] == "quota_exceeded"
        assert result["interrupt"] == {"type": "quota"}

    async def test_exact_budget_passes(self):
        # 边界：单步预估 == 剩余预算时放行
        assert (
            await bs.billing_service.check_budget(
                {"billing_context": {"remaining_budget": 1000}}, 1000
            )
            is None
        )

    async def test_no_billing_context_passes(self):
        assert await bs.billing_service.check_budget({}, 999999) is None


class TestRefundEdge:
    async def test_audit_rejected_marks_status_without_balance_refund(self, db, monkeypatch):
        refund = SysAiRefund(id=21, user_id=1, billing_id=5, amount=200, reason="误扣", status=1)
        db.add(refund)
        await db.flush()
        monkeypatch.setattr(rf.ai_refund_repository, "get_by_id", AsyncMock(return_value=refund))
        increases = []

        async def _increase(*args, **kwargs):
            increases.append(args)

        monkeypatch.setattr(rf.balance_service, "increase", _increase)

        result = await rf.refund_service.audit_refund(db, 21, False, "证据不足", 2)
        assert result.status == 3
        assert result.audit_remark == "证据不足"
        assert increases == []  # 驳回不回补余额、不写 source=refund 流水

    async def test_apply_refund_dirty_reason_preserved(self, db, monkeypatch):
        # 合法码点 emoji + 零宽字符 + CRLF + 全半角混杂 + 超长
        dirty = "误扣\r\n零宽​全角ＡＢｃ半角ABC\U0001f600" + "细节补充" * 30
        monkeypatch.setattr(
            rf.ai_billing_repository,
            "get_by_id",
            AsyncMock(return_value=SimpleNamespace(id=31, user_id=1, credits=5000)),
        )
        monkeypatch.setattr(
            rf.ai_refund_repository, "get_pending_by_billing_id", AsyncMock(return_value=None)
        )
        result = await rf.refund_service.apply_refund(db, 1, 31, 100, dirty)
        assert result.reason == dirty

    async def test_apply_refund_amount_exceeds_record_credits_rejected(self, monkeypatch):
        """API接口.md §4：退款 amount 超过可退金额应 A0400（回归保护）"""
        monkeypatch.setattr(
            rf.ai_billing_repository,
            "get_by_id",
            AsyncMock(return_value=SimpleNamespace(id=5, user_id=1, credits=100)),
        )
        monkeypatch.setattr(
            rf.ai_refund_repository, "get_pending_by_billing_id", AsyncMock(return_value=None)
        )
        monkeypatch.setattr(
            rf.ai_refund_repository,
            "create_refund",
            AsyncMock(
                return_value=SimpleNamespace(
                    id=1,
                    user_id=1,
                    billing_id=5,
                    amount=999999,
                    reason="超额",
                    status=1,
                    create_time=datetime.now(),
                )
            ),
        )
        with pytest.raises(BusinessException) as exc:
            await rf.refund_service.apply_refund(_DB, 1, 5, 999999, "超额退款")
        assert exc.value.code == ResultCode.PARAM_ERROR


class TestBillCrossMonth:
    """统计口径：月结账单按月隔离，不跨月串账"""

    async def test_monthly_bill_isolated_by_month(self, db, mock_redis):
        now = datetime.now()
        prev_month_day12 = (now.replace(day=1) - timedelta(days=1)).replace(day=12, hour=12)

        def _billing(credits):
            return SysAiBilling(
                user_id=4242,
                model="m",
                bill_type="chat",
                credits=credits,
                input_tokens=1,
                output_tokens=1,
                quota_consumed=credits,
                pre_deduct=credits,
            )

        prev_record = _billing(700)
        prev_record.create_time = prev_month_day12
        cur_record = _billing(300)
        cur_record.create_time = now
        db.add_all([prev_record, cur_record])
        await db.flush()

        prev_label = prev_month_day12.strftime("%Y-%m")
        cur_label = now.strftime("%Y-%m")
        prev_bill = await blm.bill_service.generate_monthly_bill(db, 4242, prev_label)
        cur_bill = await blm.bill_service.generate_monthly_bill(db, 4242, cur_label)
        assert prev_bill.total_consume == 700
        assert cur_bill.total_consume == 300

    async def test_bill_invalid_month_rejected(self, mock_redis):
        with pytest.raises(BusinessException) as exc:
            await blm.bill_service.get_bill(_DB, 1, "2026-13")
        assert exc.value.code == ResultCode.PARAM_ERROR
