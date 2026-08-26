"""user_repository 积分余额 CAS 乐观锁的真实 SQL 语义测试（MySQL dehaze_test）"""

from decimal import Decimal

import pytest

from app.models.entity.sys_user import SysUser
from app.repository.user_repository import user_repository

pytestmark = pytest.mark.requires_db


async def _make_user(db, **overrides):
    fields = {"username": "u1", "nickname": "n", "password": "x"}
    fields.update(overrides)
    user = SysUser(**fields)
    db.add(user)
    await db.flush()
    return user


async def test_increase_balance_cas_updates_balance_and_version(db):
    user = await _make_user(db)
    assert await user_repository.increase_balance_cas(db, user.id, Decimal("100"), 0) is True
    await db.refresh(user)
    assert user.credits_balance == Decimal("100.00")
    assert user.credits_version == 1


async def test_deduct_balance_cas_version_mismatch_fails(db):
    user = await _make_user(db, credits_version=5)
    assert await user_repository.deduct_balance_cas(db, user.id, Decimal("10"), 0) is False
    await db.refresh(user)
    assert user.credits_balance == Decimal("0.00")
    assert user.credits_version == 5


async def test_deduct_balance_cas_exact_decimal_arithmetic(db):
    """MySQL DECIMAL 精确十进制运算：0.10 累扣三次无浮点漂移"""
    user = await _make_user(db, credits_balance=Decimal("1.00"), credits_version=0)
    for version in range(3):
        assert await user_repository.deduct_balance_cas(db, user.id, Decimal("0.10"), version)
    await db.refresh(user)
    assert user.credits_balance == Decimal("0.70")
    assert user.credits_version == 3
