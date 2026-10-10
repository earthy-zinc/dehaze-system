"""user_repository 积分余额相对值 UPDATE 的真实 SQL 语义测试（MySQL dehaze_test）"""

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


async def test_increase_balance_updates_balance(db):
    user = await _make_user(db)
    assert await user_repository.increase_balance(db, user.id, Decimal("100")) is True
    await db.refresh(user)
    assert user.credits_balance == Decimal("100.00")


async def test_deduct_balance_missing_account_returns_false(db):
    """行不存在语义保留：rowcount=0 → False（供上层显式失败）"""
    assert await user_repository.deduct_balance(db, 99999999, Decimal("10")) is False


async def test_deduct_balance_exact_decimal_arithmetic(db):
    """MySQL DECIMAL 精确十进制运算：0.10 累扣三次无浮点漂移"""
    user = await _make_user(db, credits_balance=Decimal("1.00"))
    for _ in range(3):
        assert await user_repository.deduct_balance(db, user.id, Decimal("0.10")) is True
    await db.refresh(user)
    assert user.credits_balance == Decimal("0.70")


async def test_get_credits_balance_returns_none_for_missing_account(db):
    assert await user_repository.get_credits_balance(db, 99999999) is None
