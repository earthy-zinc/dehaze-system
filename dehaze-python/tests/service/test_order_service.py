"""订单服务测试：创建（vip/credit、冗余字段、防重锁/券锁定/下架）、取消与超时取消解冻。"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.cache.redis_lock import acquire_lock, release_lock
from app.models.entity.sys_order import SysOrder
from app.repository.order_repository import order_repository
from app.service.order.order_service import OrderService

pytestmark = pytest.mark.requires_db


def _pkg(**overrides):
    base = dict(
        id=1,
        name="黄金月卡",
        package_type="vip",
        level_code="level_1",
        period_days=30,
        credit_amount=None,
        status=1,
        deleted=0,
        original_price=10000,
        sale_price=9000,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _price(original_price=10000, discount_amount=1000, coupon_amount=0, payable_amount=9000):
    return {
        "originalPrice": original_price,
        "discountAmount": discount_amount,
        "couponAmount": coupon_amount,
        "payableAmount": payable_amount,
    }


def _build_service(*, pkg=None, price=None, lock_coupon=True, user_coupon_repo=None):
    pkg = pkg if pkg is not None else _pkg()
    pkg_repo = SimpleNamespace(get_by_id=AsyncMock(return_value=pkg))
    pkg_svc = SimpleNamespace(calculate_price=AsyncMock(return_value=price or _price()))
    uc_repo = user_coupon_repo or SimpleNamespace(
        lock_coupon=AsyncMock(return_value=lock_coupon),
        release_coupon=AsyncMock(return_value=True),
    )
    return OrderService(
        coupon_repository=SimpleNamespace(increment_used_qty=AsyncMock()),
        user_coupon_repository=uc_repo,
        mongo_audit_log_repository=SimpleNamespace(create_audit_async=lambda *a, **k: None),
        order_repository=order_repository,
        package_repository=pkg_repo,
        package_service=pkg_svc,
        payment_record_repository=SimpleNamespace(create=AsyncMock()),
        refund_record_repository=SimpleNamespace(),
        payment_channel_service=SimpleNamespace(close_order=AsyncMock()),
        balance_account_service=SimpleNamespace(unfreeze=AsyncMock()),
    )


async def _get_order_by_no(db, order_no):
    return await order_repository.get_by_order_no(db, order_no)


class TestCreate:
    async def test_create_vip_order(self, db):
        svc = _build_service()
        data = await svc.create(db, {"packageId": 1, "payMethod": "balance"}, 100)

        assert data["orderNo"]
        assert data["paid"] is False
        order = await _get_order_by_no(db, data["orderNo"])
        assert order.package_type == "vip"
        assert order.package_level == "level_1"
        assert order.period_days == 30
        assert order.credit_amount is None
        assert order.payable_amount == 9000
        assert order.status == 1

    async def test_create_credit_order_keeps_credit_fields(self, db):
        credit_pkg = _pkg(
            package_type="credit",
            level_code=None,
            period_days=None,
            credit_amount=10000,
            original_price=5000,
            sale_price=5000,
        )
        svc = _build_service(pkg=credit_pkg, price=_price(5000, 0, 0, 5000))
        data = await svc.create(db, {"packageId": 1, "payMethod": "balance"}, 100)

        order = await _get_order_by_no(db, data["orderNo"])
        assert order.package_type == "credit"
        assert order.package_level is None
        assert order.period_days is None
        assert order.credit_amount == 10000

    async def test_create_duplicate_raises_a0539(self, db):
        svc = _build_service()
        lock_key = "order:lock:100:1"
        token = await acquire_lock(lock_key, 10)
        try:
            with pytest.raises(BusinessException) as excinfo:
                await svc.create(db, {"packageId": 1, "payMethod": "balance"}, 100)
            assert excinfo.value.code == ResultCode.DUPLICATE_ORDER
        finally:
            await release_lock(lock_key, token)

    async def test_create_coupon_lock_failed_raises_a0525(self, db):
        svc = _build_service(lock_coupon=False)
        with pytest.raises(BusinessException) as excinfo:
            await svc.create(
                db, {"packageId": 1, "couponId": 10, "payMethod": "balance"}, 100
            )
        assert excinfo.value.code == ResultCode.COUPON_LOCK_FAILED

    async def test_create_off_shelf_package_raises_a0521(self, db):
        svc = _build_service(pkg=_pkg(status=0))
        with pytest.raises(BusinessException) as excinfo:
            await svc.create(db, {"packageId": 1, "payMethod": "balance"}, 100)
        assert excinfo.value.code == ResultCode.PACKAGE_OFF_SHELF

    async def test_create_combined_invalid_balance_amount_raises(self, db):
        svc = _build_service()
        with pytest.raises(BusinessException) as excinfo:
            await svc.create(
                db,
                {"packageId": 1, "payMethod": "combined", "balanceAmount": 0},
                100,
            )
        assert excinfo.value.code == ResultCode.PARAM_ERROR


class TestCancelExpire:
    async def test_cancel_releases_coupon_and_cancels(self, db):
        order = SysOrder(
            order_no="TEST-CANCEL-001",
            user_id=100,
            package_id=1,
            package_name="黄金月卡",
            package_type="vip",
            package_level="level_1",
            period_days=30,
            original_price=10000,
            payable_amount=9000,
            coupon_id=9,
            pay_method="wechat",
            status=1,
            expire_time=datetime.now() + timedelta(minutes=5),
            is_auto_renew=0,
        )
        await order_repository.create(db, order)
        await db.flush()

        uc_repo = SimpleNamespace(release_coupon=AsyncMock(return_value=True))
        svc = _build_service(user_coupon_repo=uc_repo)

        await svc.cancel(db, order.order_no, "测试取消", 100)
        refreshed = await _get_order_by_no(db, order.order_no)
        assert refreshed.status == 4
        assert refreshed.cancel_reason
        uc_repo.release_coupon.assert_awaited_once_with(db, 9)

    async def test_cancel_unfreezes_combined_balance(self, db):
        # pending→cancelled 用户主动取消需解冻组合支付的冻结余额（与超时取消同规则）
        order = SysOrder(
            order_no="TEST-CANCEL-CMB",
            user_id=100,
            package_id=1,
            package_name="黄金月卡",
            package_type="vip",
            package_level="level_1",
            period_days=30,
            original_price=10000,
            payable_amount=9000,
            balance_amount=3000,
            pay_method="combined",
            status=1,
            expire_time=datetime.now() + timedelta(minutes=5),
            is_auto_renew=0,
        )
        await order_repository.create(db, order)
        await db.flush()

        svc = _build_service()
        unfreeze = AsyncMock()
        svc.balance_account_service = SimpleNamespace(
            unfreeze=unfreeze,
            get_account=AsyncMock(return_value=SimpleNamespace(frozen_balance=3000)),
        )

        await svc.cancel(db, order.order_no, "测试取消", 100)
        unfreeze.assert_awaited_once_with(db, 100, 3000)
        refreshed = await _get_order_by_no(db, order.order_no)
        assert refreshed.status == 4

    async def test_cancel_without_frozen_balance_skips_unfreeze(self, db):
        # balance 支付在 pay 阶段冻结后立即扣减，pending 未支付订单从未冻结，
        # 直接取消应跳过解冻而非抛 A0500「余额解冻失败」
        order = SysOrder(
            order_no="TEST-CANCEL-NOFREEZE",
            user_id=100,
            package_id=1,
            package_name="黄金月卡",
            package_type="vip",
            package_level="level_1",
            period_days=30,
            original_price=10000,
            payable_amount=9000,
            pay_method="balance",
            status=1,
            expire_time=datetime.now() + timedelta(minutes=5),
            is_auto_renew=0,
        )
        await order_repository.create(db, order)
        await db.flush()

        svc = _build_service()
        unfreeze = AsyncMock()
        svc.balance_account_service = SimpleNamespace(
            unfreeze=unfreeze,
            get_account=AsyncMock(return_value=SimpleNamespace(frozen_balance=0)),
        )

        await svc.cancel(db, order.order_no, "测试取消", 100)
        unfreeze.assert_not_awaited()
        refreshed = await _get_order_by_no(db, order.order_no)
        assert refreshed.status == 4

    async def test_expire_orders_unfreeze_balance_for_combined(self, db):
        order = SysOrder(
            order_no="TEST-EXPIRE-001",
            user_id=100,
            package_id=1,
            package_name="黄金月卡",
            package_type="vip",
            package_level="level_1",
            period_days=30,
            original_price=10000,
            payable_amount=9000,
            balance_amount=3000,
            pay_method="combined",
            status=1,
            expire_time=datetime.now() - timedelta(minutes=1),
            is_auto_renew=0,
        )
        await order_repository.create(db, order)
        await db.flush()

        unfreeze = AsyncMock()
        svc = _build_service()
        svc.balance_account_service = SimpleNamespace(
            unfreeze=unfreeze,
            get_account=AsyncMock(return_value=SimpleNamespace(frozen_balance=3000)),
        )

        count = await svc.expire_orders(db)
        assert count == 1
        refreshed = await _get_order_by_no(db, order.order_no)
        assert refreshed.status == 4
        unfreeze.assert_awaited_once_with(db, 100, 3000)
