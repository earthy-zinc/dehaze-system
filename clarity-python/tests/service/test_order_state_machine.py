"""订单状态机/越权/支付安全/退款边界/查询与归档补充测试。

对照测试用例.md：T-OM-017a（取消/超时/退款状态拒付）、T-OM-045（已支付不可取消）、
T-OM-046/053（越权 A0530）、T-OM-052（主观原因拦截）、T-OM-057/058（折算为 0）、
T-OM-027/029（回调幂等/锁）、T-OM-024（金额快照）、T-OM-097/098（到期归档）。
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.cache.redis_lock import acquire_lock
from app.models.entity.sys_order import SysOrder
from app.repository.coupon_repository import CouponRepository, UserCouponRepository
from app.repository.mongo_audit_log_repository import MongoAuditLogRepository
from app.repository.order_repository import order_repository
from app.repository.package_repository import PackageRepository
from app.repository.payment_record_repository import PaymentRecordRepository
from app.repository.refund_record_repository import (
    RefundRecordRepository,
    refund_record_repository,
)
from app.service.order.order_service import OrderService
from app.service.order.payment_service import PaymentService
from app.service.order.refund_service import RefundService
from app.service.package_service import PackageService
from app.service.payment_channel_service import PaymentChannelService

pytestmark = pytest.mark.requires_db


async def _seed_order(db, *, order_no, user_id=100, package_type="vip", **overrides):
    base = {
        "order_no": order_no,
        "user_id": user_id,
        "package_id": 1,
        "package_name": "黄金月卡",
        "package_type": package_type,
        "package_level": "level_1" if package_type == "vip" else None,
        "period_days": 30 if package_type == "vip" else None,
        "credit_amount": None if package_type == "vip" else 10000,
        "original_price": 10000,
        "discount_amount": 1000,
        "payable_amount": 9000,
        "balance_amount": 0,
        "paid_amount": 0,
        "pay_method": None,
        "status": 1,
        "expire_time": datetime.now() + timedelta(minutes=5),
        "is_auto_renew": 0,
    }
    base.update(overrides)
    order = SysOrder(**base)
    await order_repository.create(db, order)
    await db.flush()
    return order


class _CouponRepoStub(CouponRepository):
    """测试替身：仅实现 increment_used_qty（no-op）。"""

    async def increment_used_qty(self, *args, **kwargs):
        return None


class _AuditRepoStub(MongoAuditLogRepository):
    """测试替身：仅实现 create_audit_async（no-op）。"""

    def create_audit_async(self, **kwargs):
        return None


class _PackageRepoStub(PackageRepository):
    """测试替身：仅实现 get_by_id（无套餐时返回 None）。"""

    async def get_by_id(self, *args, **kwargs):
        return None


class _PackageServiceStub(PackageService):
    """测试替身：仅实现 calculate_price（返回固定试算结果）。"""

    async def calculate_price(self, *args, **kwargs):
        return {
            "originalPrice": 10000,
            "discountAmount": 1000,
            "couponAmount": 0,
            "payableAmount": 9000,
        }


class _PaymentRecordRepoStub(PaymentRecordRepository):
    """测试替身：仅实现 create（委托注入实现）/ list_by_order_id（空）。"""

    def __init__(self, create):
        self._create = create

    async def create(self, *args, **kwargs):
        return await self._create(*args, **kwargs)

    async def list_by_order_id(self, *args, **kwargs):
        return []


class _RefundRecordRepoStub(RefundRecordRepository):
    """测试替身：仅实现 get_by_order_id（返回 None）。"""

    async def get_by_order_id(self, *args, **kwargs):
        return None


class _PaymentChannelStub(PaymentChannelService):
    """测试替身：仅实现 close_order（no-op）。"""

    def __init__(self):
        pass

    async def close_order(self, *args, **kwargs):
        return None


def _order_service(user_coupon_repo=None):
    return OrderService(
        coupon_repository=_CouponRepoStub(),
        user_coupon_repository=cast(  # 替身：用例对 release_coupon 做 AsyncMock 断言
            UserCouponRepository,
            user_coupon_repo
            or SimpleNamespace(
                lock_coupon=AsyncMock(return_value=True),
                release_coupon=AsyncMock(return_value=True),
            ),
        ),
        mongo_audit_log_repository=_AuditRepoStub(),
        order_repository=order_repository,
        package_repository=_PackageRepoStub(),
        package_service=_PackageServiceStub(),
        payment_record_repository=_PaymentRecordRepoStub(AsyncMock()),
        refund_record_repository=_RefundRecordRepoStub(),
        payment_channel_service=_PaymentChannelStub(),
        balance_account_service=SimpleNamespace(
            unfreeze=AsyncMock(),
            get_account=AsyncMock(return_value=SimpleNamespace(frozen_balance=0)),
        ),
    )


def _payment_service(**kw):
    defaults = {
        "order_repository": order_repository,
        "package_repository": SimpleNamespace(get_by_id=AsyncMock(return_value=None)),
        "payment_record_repository": SimpleNamespace(
            create=AsyncMock(),
            get_by_payment_no=AsyncMock(return_value=None),
            get_pending_by_order_id=AsyncMock(return_value=None),
        ),
        "payment_channel_service": SimpleNamespace(
            unified_order=AsyncMock(
                return_value=SimpleNamespace(pay_url="http://pay", qr_code="qr", success=True)
            ),
            verify_callback=AsyncMock(
                return_value=SimpleNamespace(
                    order_no="", amount=9000, channel_payment_no="CHAN-X", success=True, raw={}
                )
            ),
        ),
        "coupon_repository": SimpleNamespace(increment_used_qty=AsyncMock()),
        "user_coupon_repository": SimpleNamespace(
            consume_coupon=AsyncMock(), get_by_id=AsyncMock(return_value=None)
        ),
        "balance_account_service": SimpleNamespace(freeze=AsyncMock(), deduct=AsyncMock()),
        "member_service": SimpleNamespace(on_order_paid=AsyncMock()),
        "ai_balance_service": SimpleNamespace(increase=AsyncMock()),
    }
    defaults.update(kw)
    return PaymentService(**defaults)


def _refund_service(**kw):
    defaults = {
        "mongo_audit_log_repository": SimpleNamespace(create_audit_async=lambda *a, **k: None),
        "order_repository": order_repository,
        "payment_record_repository": SimpleNamespace(
            list_by_order_id=AsyncMock(
                return_value=[SimpleNamespace(channel="wechat", payment_no="PAY-REF")]
            )
        ),
        "refund_record_repository": refund_record_repository,
        "balance_refund_repository": SimpleNamespace(create=AsyncMock(), get_by_id=AsyncMock()),
        "payment_channel_service": SimpleNamespace(
            refund=AsyncMock(
                return_value=SimpleNamespace(
                    success=True, channel_refund_no="CR-1", error_message=None
                )
            )
        ),
        "balance_account_service": SimpleNamespace(refund=AsyncMock()),
        "member_service": SimpleNamespace(on_order_refunded=AsyncMock()),
        "ai_balance_service": SimpleNamespace(
            deduct=AsyncMock(), get_balance=AsyncMock(return_value=10000)
        ),
    }
    defaults.update(kw)
    return RefundService(**defaults)


async def _get_order(db, order_no):
    order = await order_repository.get_by_order_no(db, order_no)
    assert order is not None
    return order


class TestStateMachine:
    """非法状态流转拒绝：T-OM-017/017a/045/054。"""

    @pytest.mark.parametrize(
        ("order_no", "status", "expected_code"),
        [
            ("SM-C-PAID", 2, ResultCode.ORDER_STATUS_INVALID),
            ("SM-C-DONE", 3, ResultCode.ORDER_STATUS_INVALID),
            ("SM-C-CANC", 4, ResultCode.ORDER_STATUS_INVALID),
            ("SM-C-REFD", 6, ResultCode.ORDER_STATUS_INVALID),
        ],
    )
    async def test_cancel_non_pending_rejected(self, db, order_no, status, expected_code):
        await _seed_order(db, order_no=order_no, status=status)
        svc = _order_service()
        with pytest.raises(BusinessException) as excinfo:
            await svc.cancel(db, order_no, "非法取消", 100)
        assert excinfo.value.code == expected_code
        assert (await _get_order(db, order_no)).status == status

    @pytest.mark.parametrize(
        ("order_no", "status"),
        [("SM-P-CANC", 4), ("SM-P-REFG", 5), ("SM-P-REFD", 6)],
    )
    async def test_pay_cancelled_or_refund_state_rejected(self, db, order_no, status):
        await _seed_order(db, order_no=order_no, status=status)
        svc = _payment_service()
        with pytest.raises(BusinessException) as excinfo:
            await svc.pay(db, order_no, {"payMethod": "balance"}, 100)
        assert excinfo.value.code == ResultCode.ORDER_STATUS_INVALID
        assert (await _get_order(db, order_no)).status == status

    async def test_pay_completed_rejected_a0533(self, db):
        await _seed_order(db, order_no="SM-P-DONE", status=3)
        svc = _payment_service()
        with pytest.raises(BusinessException) as excinfo:
            await svc.pay(db, "SM-P-DONE", {"payMethod": "balance"}, 100)
        assert excinfo.value.code == ResultCode.ORDER_ALREADY_PAID

    async def test_pay_expired_pending_rejected_a0532(self, db):
        await _seed_order(
            db, order_no="SM-P-EXP", expire_time=datetime.now() - timedelta(minutes=1)
        )
        svc = _payment_service()
        with pytest.raises(BusinessException) as excinfo:
            await svc.pay(db, "SM-P-EXP", {"payMethod": "balance"}, 100)
        assert excinfo.value.code == ResultCode.ORDER_EXPIRED
        assert (await _get_order(db, "SM-P-EXP")).status == 1

    async def test_pay_invalid_method_rejected(self, db):
        await _seed_order(db, order_no="SM-P-BADM")
        svc = _payment_service()
        with pytest.raises(BusinessException) as excinfo:
            await svc.pay(db, "SM-P-BADM", {"payMethod": "diamond"}, 100)
        assert excinfo.value.code == ResultCode.PARAM_ERROR

    async def test_apply_refund_pending_and_cancelled_rejected(self, db):
        for order_no, status in (("SM-R-PEND", 1), ("SM-R-CANC", 4)):
            await _seed_order(db, order_no=order_no, status=status)
            svc = _refund_service()
            with pytest.raises(BusinessException) as excinfo:
                await svc.apply_refund(db, order_no, {"reasonType": "after_sale"}, 100)
            assert excinfo.value.code == ResultCode.ORDER_STATUS_INVALID
            assert (await _get_order(db, order_no)).status == status


class TestOwnershipGuard:
    """越权防护：查/取消/支付/售后他人订单一律 A0530（不泄露存在性）。"""

    async def test_cancel_other_users_order_rejected(self, db):
        await _seed_order(db, order_no="OWN-C")
        svc = _order_service()
        with pytest.raises(BusinessException) as excinfo:
            await svc.cancel(db, "OWN-C", "越权取消", 999)
        assert excinfo.value.code == ResultCode.ORDER_NOT_FOUND
        assert (await _get_order(db, "OWN-C")).status == 1

    async def test_pay_other_users_order_rejected(self, db):
        await _seed_order(db, order_no="OWN-P")
        svc = _payment_service()
        with pytest.raises(BusinessException) as excinfo:
            await svc.pay(db, "OWN-P", {"payMethod": "balance"}, 999)
        assert excinfo.value.code == ResultCode.ORDER_NOT_FOUND
        assert (await _get_order(db, "OWN-P")).status == 1

    async def test_apply_refund_other_users_order_rejected(self, db):
        await _seed_order(db, order_no="OWN-R", status=2, paid_amount=9000)
        svc = _refund_service()
        with pytest.raises(BusinessException) as excinfo:
            await svc.apply_refund(db, "OWN-R", {"reasonType": "after_sale"}, 999)
        assert excinfo.value.code == ResultCode.ORDER_NOT_FOUND
        assert (await _get_order(db, "OWN-R")).status == 2

    async def test_get_detail_other_user_rejected(self, db):
        await _seed_order(db, order_no="OWN-D")
        svc = _order_service()
        with pytest.raises(BusinessException) as excinfo:
            await svc.get_detail(db, "OWN-D", user_id=999)
        assert excinfo.value.code == ResultCode.ORDER_NOT_FOUND

    async def test_get_detail_owner_and_admin_allowed(self, db):
        await _seed_order(db, order_no="OWN-OK")
        await _seed_order(db, order_no="OWN-OK-ADMIN")
        svc = _order_service()
        owner_vo = await svc.get_detail(db, "OWN-OK", user_id=100)
        assert owner_vo["orderNo"] == "OWN-OK"
        admin_vo = await svc.get_detail(db, "OWN-OK-ADMIN", user_id=None)
        assert admin_vo["orderNo"] == "OWN-OK-ADMIN"
        # _admin 为缓存归属判定的内部标记，不得外露到 API 响应
        assert "_admin" not in owner_vo
        assert "_admin" not in admin_vo


class TestPaymentSecurity:
    """支付安全：金额快照一致性、回调幂等/状态防线。"""

    async def test_pay_uses_order_amount_snapshot_not_current_package_price(self, db):
        # 建单后套餐改价（sale_price 99999）不影响支付金额：以订单冗余 payable_amount 为准
        await _seed_order(db, order_no="PS-SNAP")
        repriced_pkg = SimpleNamespace(
            id=1, name="黄金月卡", period_days=30, sale_price=99999, status=1
        )
        svc = _payment_service(
            package_repository=SimpleNamespace(get_by_id=AsyncMock(return_value=repriced_pkg))
        )
        result = await svc.pay(db, "PS-SNAP", {"payMethod": "balance"}, 100)

        assert result["paid"] is True
        order = await _get_order(db, "PS-SNAP")
        assert order.status == 2
        assert order.paid_amount == 9000

    async def test_duplicate_callback_executes_fulfillment_once(self, db):
        await _seed_order(db, order_no="PS-IDEM")
        on_order_paid = AsyncMock()
        cb = SimpleNamespace(
            order_no="PS-IDEM", amount=9000, channel_payment_no="CHAN-IDEM", success=True, raw={}
        )
        svc = _payment_service(
            payment_channel_service=SimpleNamespace(verify_callback=AsyncMock(return_value=cb)),
            member_service=SimpleNamespace(on_order_paid=on_order_paid),
        )

        assert await svc.handle_payment_callback(db, "wechat", {}, b"") is True
        assert await svc.handle_payment_callback(db, "wechat", {}, b"") is True

        order = await _get_order(db, "PS-IDEM")
        assert order.status == 2
        assert on_order_paid.await_count == 1

    async def test_callback_for_cancelled_order_rejected(self, db):
        await _seed_order(db, order_no="PS-CB-CANC", status=4)
        cb = SimpleNamespace(
            order_no="PS-CB-CANC", amount=9000, channel_payment_no="CHAN-C1", success=True, raw={}
        )
        svc = _payment_service(
            payment_channel_service=SimpleNamespace(verify_callback=AsyncMock(return_value=cb))
        )
        assert await svc.handle_payment_callback(db, "wechat", {}, b"") is False
        assert (await _get_order(db, "PS-CB-CANC")).status == 4

    async def test_callback_with_lock_held_idempotent_success(self, db):
        # 分布式锁被并发持有 → 视为重复回调，幂等返回成功且不执行业务（T-OM-029）
        await _seed_order(db, order_no="PS-CB-LOCK")
        cb = SimpleNamespace(
            order_no="PS-CB-LOCK", amount=9000, channel_payment_no="CHAN-L1", success=True, raw={}
        )
        svc = _payment_service(
            payment_channel_service=SimpleNamespace(verify_callback=AsyncMock(return_value=cb))
        )
        lock_key = "payment:lock:PS-CB-LOCK"
        token = await acquire_lock(lock_key, 10)
        assert token is not None
        try:
            assert await svc.handle_payment_callback(db, "wechat", {}, b"") is True
        finally:
            from app.infrastructure.cache.redis_lock import release_lock

            await release_lock(lock_key, token)
        assert (await _get_order(db, "PS-CB-LOCK")).status == 1

    async def test_callback_unknown_order_no_returns_false(self, db):
        cb = SimpleNamespace(
            order_no="NO-SUCH-ORDER",
            amount=9000,
            channel_payment_no="CHAN-N1",
            success=True,
            raw={},
        )
        svc = _payment_service(
            payment_channel_service=SimpleNamespace(verify_callback=AsyncMock(return_value=cb))
        )
        assert await svc.handle_payment_callback(db, "wechat", {}, b"") is False


class TestRefundEdges:
    """退款折算边界与审核状态防线：T-OM-052/057/058/060。"""

    async def test_apply_vip_fully_used_refund_zero(self, db):
        await _seed_order(
            db,
            order_no="RE-VIP-OUT",
            status=2,
            paid_amount=9000,
            paid_time=datetime.now() - timedelta(days=40),
        )
        svc = _refund_service()
        data = await svc.apply_refund(db, "RE-VIP-OUT", {"reasonType": "after_sale"}, 100)
        assert data["refundAmount"] == 0
        refund = await refund_record_repository.get_by_order_id(
            db, (await _get_order(db, "RE-VIP-OUT")).id
        )
        assert refund is not None
        assert refund.used_days is not None
        assert refund.used_days >= 30

    async def test_apply_credit_fully_consumed_refund_zero(self, db):
        await _seed_order(
            db,
            order_no="RE-CR-OUT",
            status=2,
            package_type="credit",
            credit_amount=1000,
            payable_amount=1000,
            paid_amount=1000,
        )
        svc = _refund_service(
            ai_balance_service=SimpleNamespace(
                deduct=AsyncMock(), get_balance=AsyncMock(return_value=0)
            )
        )
        data = await svc.apply_refund(db, "RE-CR-OUT", {"reasonType": "other"}, 100)
        assert data["refundAmount"] == 0

    async def test_apply_vip_without_period_refund_zero(self, db):
        await _seed_order(db, order_no="RE-VIP-NOPD", status=2, period_days=None, paid_amount=9000)
        svc = _refund_service()
        data = await svc.apply_refund(db, "RE-VIP-NOPD", {"reasonType": "after_sale"}, 100)
        assert data["refundAmount"] == 0

    @pytest.mark.parametrize(
        ("used_days", "period", "paid"),
        [(1, 30, 9000), (10, 30, 9000), (15, 45, 12345)],
    )
    async def test_refund_amount_invariant_within_paid(self, db, used_days, period, paid):
        order_no = f"RE-INV-{used_days}-{period}"
        await _seed_order(
            db,
            order_no=order_no,
            status=2,
            period_days=period,
            payable_amount=paid,
            paid_amount=paid,
            paid_time=datetime.now() - timedelta(days=used_days),
        )
        svc = _refund_service()
        data = await svc.apply_refund(db, order_no, {"reasonType": "after_sale"}, 100)
        assert 0 <= data["refundAmount"] <= paid

    async def test_apply_subjective_reason_type_rejected(self, db):
        # 主观原因（不想要/买错了类）不是合法 reasonType，不进入售后流程
        await _seed_order(db, order_no="RE-BAD-TYPE", status=2, paid_amount=9000)
        svc = _refund_service()
        with pytest.raises(BusinessException) as excinfo:
            await svc.apply_refund(db, "RE-BAD-TYPE", {"reasonType": "no_want"}, 100)
        assert excinfo.value.code == ResultCode.PARAM_ERROR
        assert (await _get_order(db, "RE-BAD-TYPE")).status == 2

    async def test_apply_custom_reason_dirty_corpus_preserved(self, db):
        # 对抗性脏语料：emoji + 零宽字符 + CRLF + 全半角混杂，原样拼接保存
        corpus = "不想要了\U0001f600​\r\n买重复了：ＡＢＣ123"
        await _seed_order(db, order_no="RE-DIRTY", status=2, paid_amount=9000)
        svc = _refund_service()
        await svc.apply_refund(db, "RE-DIRTY", {"reasonType": "other", "customReason": corpus}, 100)
        order = await _get_order(db, "RE-DIRTY")
        refund = await refund_record_repository.get_by_order_id(db, order.id)
        assert refund is not None
        assert refund.reason == f"other:{corpus}"
        assert refund.reason_type == "other"

    async def test_approve_already_refunded_rejected(self, db):
        order = await _seed_order(db, order_no="RE-AP-DUP", status=2, paid_amount=9000)
        from app.models.entity.sys_refund_record import SysRefundRecord

        refund = SysRefundRecord(
            refund_no="REF-RE-DUP",
            order_id=order.id,
            user_id=order.user_id,
            refund_amount=6000,
            reason_type="after_sale",
            reason="已退",
            status=2,
            channel="balance",
            apply_time=datetime.now(),
            retry_count=0,
        )
        await refund_record_repository.create(db, refund)
        await db.flush()
        svc = _refund_service()
        with pytest.raises(BusinessException) as excinfo:
            await svc.approve_refund(db, refund.id, {"remark": "x"}, 200)
        assert excinfo.value.code == ResultCode.ORDER_STATUS_INVALID

    async def test_approve_refund_not_found_a0537(self, db):
        svc = _refund_service()
        with pytest.raises(BusinessException) as excinfo:
            await svc.approve_refund(db, 99999999, {"remark": "x"}, 200)
        assert excinfo.value.code == ResultCode.REFUND_NOT_FOUND


class TestQueryAndArchive:
    """我的订单隔离/筛选与到期归档：T-OM-041/042/097/098。"""

    async def test_list_my_returns_only_own_orders(self, db):
        await _seed_order(db, order_no="QRY-MINE-1", user_id=100)
        await _seed_order(db, order_no="QRY-MINE-2", user_id=100, status=2)
        await _seed_order(db, order_no="QRY-OTHER", user_id=200)
        svc = _order_service()
        data = await svc.list_my(db, 100, {"pageNum": 1, "pageSize": 10, "status": None})
        order_nos = {o["orderNo"] for o in data["list"]}
        assert data["total"] >= 2
        assert {"QRY-MINE-1", "QRY-MINE-2"} <= order_nos
        assert "QRY-OTHER" not in order_nos

    async def test_list_my_status_filter(self, db):
        await _seed_order(db, order_no="QRY-FT-P", user_id=100)
        await _seed_order(db, order_no="QRY-FT-D", user_id=100, status=2)
        svc = _order_service()
        data = await svc.list_my(db, 100, {"pageNum": 1, "pageSize": 10, "status": "paid"})
        assert data["total"] >= 1
        assert all(o["status"] == "paid" for o in data["list"])

    async def test_complete_expired_vip_orders_marks_completed(self, db):
        await _seed_order(
            db,
            order_no="ARC-VIP",
            status=2,
            paid_amount=9000,
            package_expire_time=datetime.now() - timedelta(days=1),
        )
        await _seed_order(
            db,
            order_no="ARC-VIP-ALIVE",
            status=2,
            paid_amount=9000,
            package_expire_time=datetime.now() + timedelta(days=1),
        )
        svc = _order_service()
        count = await svc.complete_expired_orders(db)
        assert count >= 1
        assert (await _get_order(db, "ARC-VIP")).status == 3
        assert (await _get_order(db, "ARC-VIP-ALIVE")).status == 2

    async def test_complete_expired_skips_credit_completed_orders(self, db):
        # 积分卡支付即 completed（status=3），不参与 paid→completed 归档
        await _seed_order(
            db,
            order_no="ARC-CR",
            status=3,
            package_type="credit",
            package_expire_time=datetime.now() - timedelta(days=1),
        )
        svc = _order_service()
        await svc.complete_expired_orders(db)
        assert (await _get_order(db, "ARC-CR")).status == 3

    async def test_expire_orders_releases_coupon(self, db):
        release_coupon = AsyncMock(return_value=True)
        svc = _order_service(user_coupon_repo=SimpleNamespace(release_coupon=release_coupon))
        await _seed_order(
            db,
            order_no="EXP-COUPON",
            coupon_id=42,
            expire_time=datetime.now() - timedelta(minutes=1),
        )
        count = await svc.expire_orders(db)
        assert count >= 1
        assert (await _get_order(db, "EXP-COUPON")).status == 4
        assert (await _get_order(db, "EXP-COUPON")).cancel_reason
        release_coupon.assert_awaited_once()
