"""平台余额账户服务测试。

覆盖 freeze/unfreeze/deduct/refund/withdraw 与乐观锁并发扣减不超卖。
使用真实 MySQL 测试库（db fixture，SAVEPOINT 回滚）+ 真实余额仓储，验证落库状态与流水。
"""

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_balance import SysBalance
from app.models.entity.sys_balance_log import SysBalanceLog
from app.service.order.balance_account_service import BalanceAccountService
from app.repository.balance_log_repository import balance_log_repository
from app.repository.balance_account_repository import balance_account_repository

pytestmark = pytest.mark.requires_db


async def _seed_account(db, user_id: int, *, balance: int = 0, frozen: int = 0, version: int = 0):
    acct = SysBalance(
        user_id=user_id,
        balance=balance,
        frozen_balance=frozen,
        version=version,
    )
    await balance_account_repository.create(db, acct)
    await db.flush()
    return acct


async def _get_account(db, user_id: int):
    return await balance_account_repository.get_by_user_id(db, user_id)


class TestFreeze:
    async def test_freeze_deducts_available_balance(self, db):
        user_id = 1001
        await _seed_account(db, user_id, balance=1000)
        svc = BalanceAccountService()
        await svc.freeze(db, user_id, 400)

        acct = await _get_account(db, user_id)
        assert acct.balance == 1000
        assert acct.frozen_balance == 400

        logs = await balance_log_repository.list_by_user(db, user_id)
        assert logs[0].change_type == "freeze"
        assert logs[0].amount == -400

    async def test_freeze_insufficient_balance_raises(self, db):
        user_id = 1002
        await _seed_account(db, user_id, balance=100)
        svc = BalanceAccountService()
        with pytest.raises(BusinessException) as excinfo:
            await svc.freeze(db, user_id, 200)
        assert excinfo.value.code == ResultCode.BALANCE_INSUFFICIENT

    async def test_freeze_zero_or_negative_amount_is_invalid(self, db):
        user_id = 1003
        await _seed_account(db, user_id, balance=100)
        svc = BalanceAccountService()
        with pytest.raises(BusinessException) as excinfo:
            await svc.freeze(db, user_id, -10)
        assert excinfo.value.code == ResultCode.PARAM_ERROR


class TestUnfreezeDeductRefund:
    async def test_unfreeze_releases_frozen(self, db):
        user_id = 2001
        await _seed_account(db, user_id, balance=1000, frozen=300)
        svc = BalanceAccountService()
        await svc.unfreeze(db, user_id, 300)

        acct = await _get_account(db, user_id)
        assert acct.balance == 1000
        assert acct.frozen_balance == 0

        logs = await balance_log_repository.list_by_user(db, user_id)
        assert logs[0].change_type == "unfreeze"

    async def test_deduct_reduces_balance_and_frozen(self, db):
        user_id = 2002
        await _seed_account(db, user_id, balance=1000, frozen=300)
        svc = BalanceAccountService()
        await svc.deduct(db, user_id, 300)

        acct = await _get_account(db, user_id)
        assert acct.balance == 700
        assert acct.frozen_balance == 0

        logs = await balance_log_repository.list_by_user(db, user_id)
        assert logs[0].change_type == "consume"

    async def test_refund_returns_balance(self, db):
        user_id = 2003
        await _seed_account(db, user_id, balance=1000)
        svc = BalanceAccountService()
        await svc.refund(db, user_id, 200)

        acct = await _get_account(db, user_id)
        assert acct.balance == 1200

        logs = await balance_log_repository.list_by_user(db, user_id)
        assert logs[0].change_type == "refund"
        assert logs[0].amount == 200

    async def test_withdraw_reduces_available_balance(self, db):
        user_id = 2004
        await _seed_account(db, user_id, balance=1000)
        svc = BalanceAccountService()
        await svc.withdraw(db, user_id, 250)

        acct = await _get_account(db, user_id)
        assert acct.balance == 750

        logs = await balance_log_repository.list_by_user(db, user_id)
        assert logs[0].change_type == "refund"
        assert logs[0].amount == -250

    async def test_withdraw_insufficient_raises(self, db):
        user_id = 2005
        await _seed_account(db, user_id, balance=100)
        svc = BalanceAccountService()
        with pytest.raises(BusinessException) as excinfo:
            await svc.withdraw(db, user_id, 500)
        assert excinfo.value.code == ResultCode.BALANCE_INSUFFICIENT


class TestConcurrentDeduct:
    async def test_stale_version_cas_rejected(self, db):
        """乐观锁 CAS 不超卖：同一 version 发起两次扣减，仅第一次生效（version 递增），
        第二次因版本已过期被拒绝，余额不被重复扣减。"""
        user_id = 3001
        await _seed_account(db, user_id, balance=1000, frozen=200)
        acct = await _get_account(db, user_id)

        ok = await balance_account_repository.deduct(db, user_id, 150, acct.version)
        assert ok is True
        second = await balance_account_repository.deduct(db, user_id, 150, acct.version)
        assert second is False

        refreshed = await _get_account(db, user_id)
        assert refreshed.balance == 850
        assert refreshed.frozen_balance == 50

    async def test_cas_prevents_oversell_beyond_frozen(self, db):
        """扣减超过冻结余额的 CAS 不生效：冻结 100，一次性扣 150 应失败。"""
        user_id = 3002
        await _seed_account(db, user_id, balance=100, frozen=100)
        acct = await _get_account(db, user_id)
        ok = await balance_account_repository.deduct(db, user_id, 150, acct.version)
        assert ok is False
        refreshed = await _get_account(db, user_id)
        assert refreshed.balance == 100
        assert refreshed.frozen_balance == 100
