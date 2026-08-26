"""自动续费服务测试：配置 next_renew_time NULL、balance 直扣、wechat 半自动、重试/关闭。"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sqlalchemy import select

from app.models.entity.sys_auto_renew import SysAutoRenew
from app.models.entity.sys_order import SysOrder
from app.repository.auto_renew_repository import auto_renew_repository
from app.repository.order_repository import order_repository
from app.service.order.auto_renew_service import AutoRenewService

pytestmark = pytest.mark.requires_db


def _pkg(**overrides):
    base = dict(
        id=1,
        name="黄金月卡",
        package_type="vip",
        level_code="level_1",
        period_days=30,
        status=1,
        deleted=0,
        original_price=10000,
        sale_price=9000,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _build_service(**kw):
    defaults = dict(
        auto_renew_repository=auto_renew_repository,
        package_repository=SimpleNamespace(get_by_id=AsyncMock(return_value=_pkg())),
        order_repository=order_repository,
        payment_service=SimpleNamespace(complete_payment=AsyncMock()),
        balance_account_service=SimpleNamespace(freeze=AsyncMock()),
    )
    defaults.update(kw)
    return AutoRenewService(**defaults)


async def _get_config(db, user_id, package_id):
    return await auto_renew_repository.get_by_user_and_package(db, user_id, package_id)


class TestConfig:
    async def test_enable_config_sets_next_renew_time_null(self, db):
        svc = _build_service()
        await svc.update_config(db, {"packageId": 1, "payMethod": "balance", "enabled": True}, 100)

        config = await _get_config(db, 100, 1)
        assert config is not None
        assert config.status == 1
        assert config.next_renew_time is None  # 首次开启不触发扣款

    async def test_credit_package_rejected(self, db):
        from app.core.exceptions import BusinessException

        svc = _build_service(
            package_repository=SimpleNamespace(
                get_by_id=AsyncMock(return_value=_pkg(package_type="credit"))
            )
        )
        with pytest.raises(BusinessException):
            await svc.update_config(
                db, {"packageId": 1, "payMethod": "balance", "enabled": True}, 100
            )


class TestExecuteRenewal:
    async def _seed_due_config(
        self, db, *, user_id=100, fail_count=0, status=1, pay_method="balance"
    ):
        config = SysAutoRenew(
            user_id=user_id,
            package_id=1,
            pay_method=pay_method,
            status=status,
            next_renew_time=datetime.now() - timedelta(minutes=1),
            fail_count=fail_count,
        )
        await auto_renew_repository.create(db, config)
        await db.flush()
        return config

    async def test_balance_renewal_direct_deduct(self, db):
        config = await self._seed_due_config(db)
        freeze = AsyncMock()
        complete_payment = AsyncMock(return_value=None)
        payment_svc = SimpleNamespace(complete_payment=complete_payment)
        svc = _build_service(
            balance_account_service=SimpleNamespace(freeze=freeze),
            payment_service=payment_svc,
        )

        count = await svc.execute_renewal(db)

        assert count == 1
        freeze.assert_awaited_once_with(db, 100, 8550)  # 9000 * 0.95 = 8550
        complete_payment.assert_awaited_once()
        refreshed = await _get_config(db, 100, 1)
        assert refreshed.fail_count == 0
        assert refreshed.next_renew_time is not None

    async def test_wechat_half_auto_creates_pending_order(self, db):
        await self._seed_due_config(db, user_id=101, pay_method="wechat")
        svc = _build_service(
            package_repository=SimpleNamespace(
                get_by_id=AsyncMock(return_value=_pkg(id=1, package_type="vip"))
            ),
        )
        count = await svc.execute_renewal(db)

        assert count == 0  # 半自动不即时支付
        stmt = select(SysOrder).where(SysOrder.is_auto_renew == 1)
        result = await db.execute(stmt)
        pending_order = result.scalar_one_or_none()
        assert pending_order is not None
        assert pending_order.status == 1
        refreshed = await _get_config(db, 101, 1)
        assert refreshed.next_renew_time is not None

    async def test_fail_max_closes_config(self, db):
        config = await self._seed_due_config(db, user_id=102, fail_count=3, status=1)
        svc = _build_service()
        await svc.execute_renewal(db)

        refreshed = await _get_config(db, 102, 1)
        assert refreshed.status == 0
        assert refreshed.close_reason
