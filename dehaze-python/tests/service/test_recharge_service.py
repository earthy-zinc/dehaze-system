"""余额充值服务测试：下单（渠道参数/非法支付方式）、回调入账（金额校验/幂等/余额流水）。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_recharge import SysRecharge
from app.repository.balance_account_repository import balance_account_repository
from app.repository.balance_log_repository import balance_log_repository
from app.repository.recharge_repository import recharge_repository
from app.service.order.balance_account_service import BalanceAccountService
from app.service.order.recharge_service import RechargeService

pytestmark = pytest.mark.requires_db


def _pay_result(pay_url="https://mock-pay.example.com/rc"):
    return SimpleNamespace(pay_url=pay_url, qr_code=pay_url, channel_order_no="MOCK-CH")


def _callback(order_no: str, amount: int, payment_no="CH-PAY-001"):
    return SimpleNamespace(order_no=order_no, amount=amount, channel_payment_no=payment_no, raw={})


def _build_service(channel_service=None, balance_account_service=None):
    return RechargeService(
        recharge_repository=recharge_repository,
        payment_channel_service=channel_service
        or SimpleNamespace(unified_order=AsyncMock(return_value=_pay_result())),
        mongo_audit_log_repository=SimpleNamespace(create_audit_async=lambda *a, **k: None),
        balance_account_service=balance_account_service or BalanceAccountService(),
    )


async def _seed_recharge(db, recharge_no: str, user_id: int = 100, amount: int = 5000, **overrides):
    record = SysRecharge(
        recharge_no=recharge_no,
        user_id=user_id,
        amount=amount,
        pay_method="wechat",
        status=1,
    )
    for k, v in overrides.items():
        setattr(record, k, v)
    await recharge_repository.create(db, record)
    await db.flush()
    return record


class TestCreateRecharge:
    async def test_create_returns_pay_params(self, db):
        svc = _build_service()
        data = await svc.create_recharge(db, {"amount": 5000, "payMethod": "wechat"}, 100)

        assert data["rechargeNo"].startswith("RC")
        assert data["payUrl"]
        record = await recharge_repository.get_by_recharge_no(db, data["rechargeNo"])
        assert record.status == 1
        assert record.amount == 5000

    async def test_create_invalid_pay_method_raises(self, db):
        svc = _build_service()
        with pytest.raises(BusinessException) as excinfo:
            await svc.create_recharge(db, {"amount": 5000, "payMethod": "balance"}, 100)
        assert excinfo.value.code == ResultCode.PARAM_ERROR


class TestRechargeCallback:
    async def test_callback_credits_balance_with_recharge_log(self, db):
        await _seed_recharge(db, "RC-OK-1")
        await balance_account_repository.get_or_create(db, 100)
        svc = _build_service()

        ok = await svc.handle_payment_callback(db, _callback("RC-OK-1", 5000), "wechat")
        assert ok is True

        record = await recharge_repository.get_by_recharge_no(db, "RC-OK-1")
        assert record.status == 2
        assert record.channel_payment_no == "CH-PAY-001"
        account = await balance_account_repository.get_by_user_id(db, 100)
        assert account.balance == 5000
        logs = await balance_log_repository.list_by_user(db, 100)
        assert logs[0].change_type == "recharge"
        assert logs[0].amount == 5000

    async def test_callback_amount_mismatch_raises_a0538(self, db):
        await _seed_recharge(db, "RC-AMT-1")
        svc = _build_service()

        with pytest.raises(BusinessException) as excinfo:
            await svc.handle_payment_callback(db, _callback("RC-AMT-1", 4000), "wechat")
        assert excinfo.value.code == ResultCode.PAYMENT_AMOUNT_MISMATCH

    async def test_callback_idempotent_no_double_credit(self, db):
        await _seed_recharge(db, "RC-IDEM-1")
        await balance_account_repository.get_or_create(db, 100)
        svc = _build_service()

        assert await svc.handle_payment_callback(db, _callback("RC-IDEM-1", 5000), "wechat") is True
        # 重复回调（不同渠道流水号）幂等返回成功，余额只入账一次
        second = await svc.handle_payment_callback(
            db, _callback("RC-IDEM-1", 5000, payment_no="CH-PAY-002"), "wechat"
        )
        assert second is True
        account = await balance_account_repository.get_by_user_id(db, 100)
        assert account.balance == 5000

    async def test_callback_unknown_recharge_no_returns_false(self, db):
        svc = _build_service()
        ok = await svc.handle_payment_callback(db, _callback("RC-NONE", 5000), "wechat")
        assert ok is False

    async def test_callback_on_closed_recharge_rejected(self, db):
        await _seed_recharge(db, "RC-CLOSED", status=3)
        await balance_account_repository.get_or_create(db, 100)
        svc = _build_service()

        ok = await svc.handle_payment_callback(db, _callback("RC-CLOSED", 5000), "wechat")
        assert ok is False
        account = await balance_account_repository.get_by_user_id(db, 100)
        assert account.balance == 0
