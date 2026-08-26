"""支付服务测试：余额/组合支付、支付回调校验/幂等/履约分流、优惠券核销。"""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_order import SysOrder
from app.repository.order_repository import order_repository
from app.service.order.payment_service import PaymentService

pytestmark = pytest.mark.requires_db


def _pay_result(pay_url="http://pay", qr_code="qr"):
    return SimpleNamespace(pay_url=pay_url, qr_code=qr_code, success=True)


def _callback(order_no, amount, payment_no, success=True, raw=None):
    return SimpleNamespace(
        order_no=order_no,
        amount=amount,
        channel_payment_no=payment_no,
        success=success,
        raw=raw or {"out_trade_no": order_no},
    )


async def _seed_order(db, *, order_no="PAY-001", user_id=100, package_type="vip", **overrides):
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
        paid_amount=0,
        pay_method=None,
        status=1,
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
        order_repository=order_repository,
        package_repository=SimpleNamespace(get_by_id=AsyncMock(return_value=None)),
        payment_record_repository=SimpleNamespace(
            create=AsyncMock(), get_by_payment_no=AsyncMock(return_value=None)
        ),
        payment_channel_service=SimpleNamespace(
            unified_order=AsyncMock(return_value=_pay_result()),
            verify_callback=AsyncMock(return_value=_callback("PAY-001", 9000, "CHAN-001")),
        ),
        coupon_repository=SimpleNamespace(increment_used_qty=AsyncMock()),
        user_coupon_repository=SimpleNamespace(
            consume_coupon=AsyncMock(), get_by_id=AsyncMock(return_value=None)
        ),
        balance_account_service=SimpleNamespace(freeze=AsyncMock(), deduct=AsyncMock()),
        member_service=SimpleNamespace(on_order_paid=AsyncMock()),
        ai_balance_service=SimpleNamespace(increase=AsyncMock()),
    )
    defaults.update(kw)
    return PaymentService(**defaults)


async def _get_order(db, order_no):
    return await order_repository.get_by_order_no(db, order_no)


class TestBalancePay:
    async def test_balance_pay_freezes_then_completes(self, db):
        await _seed_order(db, order_no="PAY-BAL", user_id=100)
        freeze = AsyncMock()
        deduct = AsyncMock()
        svc = _build_service(
            balance_account_service=SimpleNamespace(freeze=freeze, deduct=deduct),
            member_service=SimpleNamespace(on_order_paid=AsyncMock()),
        )

        result = await svc.pay(db, "PAY-BAL", {"payMethod": "balance"}, 100)

        assert result["paid"] is True
        freeze.assert_awaited_once_with(db, 100, 9000)
        deduct.assert_awaited_once_with(db, 100, 9000)
        order = await _get_order(db, "PAY-BAL")
        assert order.status == 2
        assert order.paid_amount == 9000

    async def test_balance_pay_insufficient_raises(self, db):
        await _seed_order(db, order_no="PAY-INS", user_id=100)

        def _raise(*a, **k):
            raise BusinessException(ResultCode.BALANCE_INSUFFICIENT)

        svc = _build_service(
            balance_account_service=SimpleNamespace(freeze=_raise, deduct=AsyncMock())
        )
        with pytest.raises(BusinessException) as excinfo:
            await svc.pay(db, "PAY-INS", {"payMethod": "balance"}, 100)
        assert excinfo.value.code == ResultCode.BALANCE_INSUFFICIENT
        order = await _get_order(db, "PAY-INS")
        assert order.status == 1

    async def test_pay_already_paid_raises(self, db):
        await _seed_order(db, order_no="PAY-DONE", status=2)
        svc = _build_service()
        with pytest.raises(BusinessException) as excinfo:
            await svc.pay(db, "PAY-DONE", {"payMethod": "balance"}, 100)
        assert excinfo.value.code == ResultCode.ORDER_ALREADY_PAID


class TestCombinedPay:
    async def test_combined_pay_freeze_and_channel_unified(self, db):
        await _seed_order(db, order_no="PAY-CMB", balance_amount=3000)
        freeze = AsyncMock()
        unified = AsyncMock(return_value=_pay_result())
        svc = _build_service(
            balance_account_service=SimpleNamespace(freeze=freeze, deduct=AsyncMock()),
            payment_channel_service=SimpleNamespace(
                unified_order=unified, verify_callback=AsyncMock()
            ),
        )

        result = await svc.pay(db, "PAY-CMB", {"payMethod": "combined"}, 100)

        assert result["paid"] is False
        freeze.assert_awaited_once_with(db, 100, 3000)
        unified.assert_awaited_once_with("combined", "PAY-CMB", 6000, "黄金月卡")
        order = await _get_order(db, "PAY-CMB")
        assert order.pay_method == "combined"
        assert order.balance_amount == 3000


class TestPaymentCallback:
    async def test_callback_amount_mismatch_raises_a0538(self, db):
        await _seed_order(db, order_no="PAY-CB1")
        cb = _callback("PAY-CB1", 5000, "CHAN-1")
        svc = _build_service(
            payment_channel_service=SimpleNamespace(verify_callback=AsyncMock(return_value=cb))
        )
        with pytest.raises(BusinessException) as excinfo:
            await svc.handle_payment_callback(db, "wechat", {}, b"")
        assert excinfo.value.code == ResultCode.PAYMENT_AMOUNT_MISMATCH

    async def test_callback_idempotent_when_already_paid(self, db):
        await _seed_order(db, order_no="PAY-CB2", status=2)
        cb = _callback("PAY-CB2", 9000, "CHAN-2")
        svc = _build_service(
            payment_channel_service=SimpleNamespace(verify_callback=AsyncMock(return_value=cb))
        )
        result = await svc.handle_payment_callback(db, "wechat", {}, b"")
        assert result is True

    async def test_callback_vip_triggers_on_order_paid(self, db):
        await _seed_order(db, order_no="PAY-CB3", package_type="vip")
        cb = _callback("PAY-CB3", 9000, "CHAN-3")
        on_order_paid = AsyncMock()
        svc = _build_service(
            payment_channel_service=SimpleNamespace(verify_callback=AsyncMock(return_value=cb)),
            member_service=SimpleNamespace(on_order_paid=on_order_paid),
        )
        result = await svc.handle_payment_callback(db, "wechat", {}, b"")
        assert result is True
        on_order_paid.assert_awaited_once()
        order = await _get_order(db, "PAY-CB3")
        assert order.status == 2

    async def test_callback_credit_credits_and_completed(self, db):
        await _seed_order(db, order_no="PAY-CB4", package_type="credit", payable_amount=5000)
        cb = _callback("PAY-CB4", 5000, "CHAN-4")
        increase = AsyncMock()
        svc = _build_service(
            payment_channel_service=SimpleNamespace(verify_callback=AsyncMock(return_value=cb)),
            ai_balance_service=SimpleNamespace(increase=increase),
            member_service=SimpleNamespace(on_order_paid=AsyncMock()),
        )
        await svc.handle_payment_callback(db, "alipay", {}, b"")
        increase.assert_awaited_once()
        order = await _get_order(db, "PAY-CB4")
        assert order.status == 3

    async def test_callback_consumes_coupon(self, db):
        await _seed_order(db, order_no="PAY-CB5", coupon_id=7)
        cb = _callback("PAY-CB5", 9000, "CHAN-5")
        consume_coupon = AsyncMock()
        increment_used_qty = AsyncMock()
        uc_repo = SimpleNamespace(
            consume_coupon=consume_coupon, get_by_id=AsyncMock(return_value=SimpleNamespace(coupon_id=3))
        )
        svc = _build_service(
            payment_channel_service=SimpleNamespace(verify_callback=AsyncMock(return_value=cb)),
            user_coupon_repository=uc_repo,
            coupon_repository=SimpleNamespace(increment_used_qty=increment_used_qty),
            member_service=SimpleNamespace(on_order_paid=AsyncMock()),
        )
        await svc.handle_payment_callback(db, "wechat", {}, b"")
        consume_coupon.assert_awaited_once()
        increment_used_qty.assert_awaited_once()
