"""退款服务测试：申请（原因类型/按商品折算/重复）、审核（按渠道回充/回退履约/失败恢复）、余额退款流程。"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_refund_record import SysRefundRecord
from app.models.entity.sys_order import SysOrder
from app.repository.balance_refund_repository import balance_refund_repository
from app.repository.order_repository import order_repository
from app.repository.refund_record_repository import refund_record_repository
from app.service.order.refund_service import RefundService

pytestmark = pytest.mark.requires_db


def _refund_result(success=True, channel_refund_no="CR-1", error_message=None):
    return SimpleNamespace(
        success=success,
        channel_refund_no=channel_refund_no,
        error_message=error_message,
    )


async def _seed_order(db, *, order_no="RF-001", user_id=100, package_type="vip", **overrides):
    base = dict(
        order_no=order_no,
        user_id=user_id,
        package_id=1,
        package_name="黄金月卡",
        package_type=package_type,
        package_level="level_1" if package_type == "vip" else None,
        period_days=30 if package_type == "vip" else None,
        credit_amount=None if package_type == "vip" else 10000,
        original_price=10000,
        discount_amount=1000,
        payable_amount=9000,
        balance_amount=0,
        paid_amount=9000,
        pay_method="balance",
        status=2,
        paid_time=datetime.now() - timedelta(days=10),
        expire_time=datetime.now() + timedelta(minutes=5),
        is_auto_renew=0,
    )
    base.update(overrides)
    order = SysOrder(**base)
    await order_repository.create(db, order)
    await db.flush()
    return order


def _build_service(**kw):
    defaults = dict(
        mongo_audit_log_repository=SimpleNamespace(create_audit_async=lambda *a, **k: None),
        order_repository=order_repository,
        payment_record_repository=SimpleNamespace(
            list_by_order_id=AsyncMock(
                return_value=[SimpleNamespace(channel="wechat", payment_no="PAY-REF")]
            )
        ),
        refund_record_repository=refund_record_repository,
        balance_refund_repository=SimpleNamespace(create=AsyncMock(), get_by_id=AsyncMock()),
        payment_channel_service=SimpleNamespace(
            refund=AsyncMock(return_value=_refund_result())
        ),
        balance_account_service=SimpleNamespace(refund=AsyncMock()),
        member_service=SimpleNamespace(on_order_refunded=AsyncMock()),
        ai_balance_service=SimpleNamespace(
            deduct=AsyncMock(), get_balance=AsyncMock(return_value=10000)
        ),
    )
    defaults.update(kw)
    return RefundService(**defaults)


class TestApplyRefund:
    async def test_apply_missing_reason_type_raises(self, db):
        await _seed_order(db, order_no="RF-A1")
        svc = _build_service()
        with pytest.raises(BusinessException) as excinfo:
            await svc.apply_refund(db, "RF-A1", {"reason": "不好用"}, 100)
        assert excinfo.value.code == ResultCode.PARAM_ERROR

    async def test_apply_vip_prorated_by_days(self, db):
        # 已用约 10 天（9天23小时 → ceil=10），剩余 20 天，退 9000 * 20/30 = 6000
        await _seed_order(
            db,
            order_no="RF-VIP",
            paid_time=datetime.now() - timedelta(days=9, hours=23),
        )
        svc = _build_service()
        data = await svc.apply_refund(
            db, "RF-VIP", {"reasonType": "after_sale", "reason": "不满意"}, 100
        )
        assert data["refundAmount"] == 6000
        refund = await _get_refund_by_no(db, data["refundNo"])
        assert refund.reason_type == "after_sale"
        assert refund.used_days == 10
        order = await order_repository.get_by_order_no(db, "RF-VIP")
        assert order.status == 5

    async def test_apply_credit_prorated_by_usage(self, db):
        # 积分余额充足（>= 到账积分）→ 未消耗，全额折算
        await _seed_order(
            db, order_no="RF-CR", package_type="credit", payable_amount=5000, paid_amount=5000
        )
        svc = _build_service()
        data = await svc.apply_refund(db, "RF-CR", {"reasonType": "other"}, 100)
        assert data["refundAmount"] == 5000
        refund = await _get_refund_by_no(db, data["refundNo"])
        assert refund.used_credits == 0

    async def test_apply_credit_partial_consumed(self, db):
        # 积分卡到账 1000，当前积分余额 600 → 缺口 400 计为本单已消耗，退 600/1000 比例
        await _seed_order(
            db,
            order_no="RF-CR2",
            package_type="credit",
            credit_amount=1000,
            payable_amount=1000,
            paid_amount=1000,
        )
        svc = _build_service(
            ai_balance_service=SimpleNamespace(
                deduct=AsyncMock(), get_balance=AsyncMock(return_value=600)
            )
        )
        data = await svc.apply_refund(db, "RF-CR2", {"reasonType": "other"}, 100)
        assert data["refundAmount"] == 600
        refund = await _get_refund_by_no(db, data["refundNo"])
        assert refund.used_credits == 400

    async def test_apply_merchant_full_refund(self, db):
        await _seed_order(db, order_no="RF-MER")
        svc = _build_service()
        data = await svc.apply_refund(db, "RF-MER", {"reasonType": "merchant"}, 100)
        assert data["refundAmount"] == 9000

    async def test_apply_duplicate_raises_a053a(self, db):
        order = await _seed_order(db, order_no="RF-DUP")
        # 订单仍为可退款状态，但已存在售后记录 → 命中唯一性校验
        await refund_record_repository.create(
            db,
            SysRefundRecord(
                refund_no="REF-DUP",
                order_id=order.id,
                user_id=order.user_id,
                refund_amount=9000,
                reason_type="after_sale",
                reason="已申请",
                status=1,
                channel="balance",
                apply_time=datetime.now(),
                retry_count=0,
            ),
        )
        await db.flush()
        svc = _build_service()
        with pytest.raises(BusinessException) as excinfo:
            await svc.apply_refund(db, "RF-DUP", {"reasonType": "after_sale"}, 100)
        assert excinfo.value.code == ResultCode.REFUND_ALREADY_EXISTS


class TestApproveRefund:
    async def _seed_refund(self, db, order, amount=9000, **overrides):
        refund = SysRefundRecord(
            refund_no="REF-001",
            order_id=order.id,
            user_id=order.user_id,
            refund_amount=amount,
            reason_type="after_sale",
            reason="测试",
            status=1,
            channel=order.pay_method,
            apply_time=datetime.now(),
            retry_count=0,
        )
        for k, v in overrides.items():
            setattr(refund, k, v)
        await refund_record_repository.create(db, refund)
        await db.flush()
        return refund

    async def test_approve_balance_refunds(self, db):
        order = await _seed_order(db, order_no="RF-AP-BAL")
        refund = await self._seed_refund(db, order, amount=6000)
        balance_refund = AsyncMock()
        svc = _build_service(
            balance_account_service=SimpleNamespace(refund=balance_refund),
            member_service=SimpleNamespace(on_order_refunded=AsyncMock()),
        )
        await svc.approve_refund(db, refund.id, {"remark": "同意"}, 200)
        balance_refund.assert_awaited_once_with(db, 100, 6000)
        refreshed = await refund_record_repository.get_by_id(db, refund.id)
        assert refreshed.status == 2
        order_ref = await order_repository.get_by_id(db, order.id)
        assert order_ref.status == 6

    async def test_approve_combined_splits(self, db):
        order = await _seed_order(
            db, order_no="RF-AP-CMB", pay_method="combined", balance_amount=3000, paid_amount=9000
        )
        refund = await self._seed_refund(db, order, amount=9000)
        balance_refund = AsyncMock()
        channel_refund = AsyncMock(return_value=_refund_result())
        payment_record_repo = SimpleNamespace(
            list_by_order_id=AsyncMock(
                return_value=[SimpleNamespace(channel="wechat", payment_no="PAY-CMB")]
            )
        )
        svc = _build_service(
            balance_account_service=SimpleNamespace(refund=balance_refund),
            payment_record_repository=payment_record_repo,
            payment_channel_service=SimpleNamespace(refund=channel_refund),
            member_service=SimpleNamespace(on_order_refunded=AsyncMock()),
        )
        await svc.approve_refund(db, refund.id, {"remark": "同意"}, 200)
        # 余额部分 = 9000 * 3000/9000 = 3000
        balance_refund.assert_awaited_once_with(db, 100, 3000)
        channel_refund.assert_awaited_once()

    async def test_approve_vip_rolls_back_fulfillment(self, db):
        order = await _seed_order(db, order_no="RF-AP-VIP", pay_method="wechat")
        refund = await self._seed_refund(db, order, amount=6000)
        on_order_refunded = AsyncMock()
        svc = _build_service(
            payment_channel_service=SimpleNamespace(refund=AsyncMock(return_value=_refund_result())),
            member_service=SimpleNamespace(on_order_refunded=on_order_refunded),
        )
        await svc.approve_refund(db, refund.id, {"remark": "同意"}, 200)
        on_order_refunded.assert_awaited_once()
        refund_ref = await refund_record_repository.get_by_id(db, refund.id)
        assert refund_ref.status == 2

    async def test_approve_failure_restores_order(self, db):
        order = await _seed_order(db, order_no="RF-AP-FAIL", pay_method="wechat")
        refund = await self._seed_refund(db, order, amount=6000)
        svc = _build_service(
            payment_channel_service=SimpleNamespace(
                refund=AsyncMock(return_value=_refund_result(success=False, error_message="渠道失败"))
            )
        )
        await svc.approve_refund(db, refund.id, {"remark": "同意"}, 200)
        refund_ref = await refund_record_repository.get_by_id(db, refund.id)
        assert refund_ref.status == 3
        assert refund_ref.error_message
        order_ref = await order_repository.get_by_id(db, order.id)
        assert order_ref.status in (2, 3)


class TestBalanceRefund:
    async def test_apply_balance_refund_returns_refund_no(self, db):
        svc = _build_service(
            balance_account_service=SimpleNamespace(
                get_balance=AsyncMock(return_value={"balance": 5000, "frozenBalance": 0})
            ),
            balance_refund_repository=balance_refund_repository,
        )
        data = await svc.apply_balance_refund(db, 100, {"amount": 5000})
        assert data["refundNo"]
        assert data["amount"] == 5000
        record = await balance_refund_repository.get_by_refund_no(db, data["refundNo"])
        assert record.status == 1

    async def test_approve_balance_refund_withdraws(self, db):
        record = SimpleNamespace(
            id=1, user_id=100, amount=5000, status=1, refund_no="BR-1", channel=None
        )
        balance_refund_repo = SimpleNamespace(
            get_by_id=AsyncMock(return_value=record),
        )
        withdraw = AsyncMock()
        svc = _build_service(
            balance_refund_repository=balance_refund_repo,
            balance_account_service=SimpleNamespace(
                get_account=AsyncMock(
                    return_value=SimpleNamespace(balance=5000, frozen_balance=0)
                ),
                withdraw=withdraw,
            ),
        )
        await svc.approve_balance_refund(db, 1, {"remark": "同意"}, 200)
        withdraw.assert_awaited_once_with(db, 100, 5000)
        assert record.status == 2


async def _get_refund_by_no(db, refund_no):
    from sqlalchemy import select

    stmt = select(SysRefundRecord).where(SysRefundRecord.refund_no == refund_no)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()
