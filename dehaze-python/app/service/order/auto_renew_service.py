"""自动续费域：续费配置管理（查询/修改）与到期代扣执行（仅会员卡）。

- 首次开启不触发扣款：配置开启时 next_renew_time 置 NULL，需购买/续费成功后才有扣款时间。
- balance：直接冻结扣减 → 会员激活 → 创建已支付续费订单。
- wechat/alipay（半自动）：到期创建【待支付】续费订单，用户主动支付；未支付则订单超时取消，
  下轮扫描计一次失败，2 小时重试新单，达 3 次关闭配置。
"""

import logging
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_auto_renew import SysAutoRenew
from app.models.entity.sys_order import SysOrder
from app.repository.auto_renew_repository import auto_renew_repository
from app.repository.order_repository import order_repository
from app.repository.package_repository import package_repository
from app.service.order.order_service import (
    ORDER_EXPIRE_MINUTES,
    _format_dt,
    _gen_order_no,
)
from app.service.order.payment_service import PaymentService

logger = logging.getLogger(__name__)


class AutoRenewService:
    def __init__(
        self,
        auto_renew_repository=auto_renew_repository,
        package_repository=package_repository,
        order_repository=order_repository,
        payment_service=None,
        balance_account_service=None,
    ):
        self.auto_renew_repository = auto_renew_repository
        self.package_repository = package_repository
        self.order_repository = order_repository
        if payment_service is None:
            from app.service.order.payment_service import payment_service as _ps

            payment_service = _ps
        self.payment_service = payment_service
        if balance_account_service is None:
            from app.service.order.balance_account_service import (
                balance_account_service as _b,
            )

            balance_account_service = _b
        self.balance_account_service = balance_account_service

    async def update_config(self, db: AsyncSession, form: dict, user_id: int) -> None:
        package_id = form["packageId"]
        pkg = await self.package_repository.get_by_id(db, package_id)
        if not pkg:
            raise BusinessException(ResultCode.PACKAGE_NOT_FOUND)
        if pkg.package_type != "vip":
            raise BusinessException(ResultCode.BUSINESS_ERROR, "积分卡不支持自动续费")

        pay_method = form["payMethod"]
        enabled = form["enabled"]
        status = 1 if enabled else 0

        # 首次开启不触发扣款：next_renew_time 置 NULL
        await self.auto_renew_repository.upsert_by_user_and_package(
            db,
            user_id,
            package_id,
            pay_method,
            status,
            next_renew_time=None,
            fail_count=0,
        )
        if not enabled:
            from sqlalchemy import update as update_stmt

            await db.execute(
                update_stmt(SysAutoRenew)
                .where(
                    SysAutoRenew.user_id == user_id,
                    SysAutoRenew.package_id == package_id,
                    SysAutoRenew.deleted == 0,
                )
                .values(close_reason="用户关闭")
            )

    async def get_config(self, db: AsyncSession, package_id: int, user_id: int) -> dict:
        pkg = await self.package_repository.get_by_id(db, package_id)
        if not pkg:
            raise BusinessException(ResultCode.PACKAGE_NOT_FOUND)

        config = await self.auto_renew_repository.get_by_user_and_package(db, user_id, package_id)
        if not config:
            return {
                "userId": user_id,
                "packageId": package_id,
                "packageName": pkg.name,
                "payMethod": "balance",
                "enabled": False,
                "failCount": 0,
                "closeReason": None,
            }

        return {
            "userId": config.user_id,
            "packageId": config.package_id,
            "packageName": pkg.name,
            "payMethod": config.pay_method,
            "enabled": config.status == 1,
            "nextRenewTime": _format_dt(config.next_renew_time),
            "failCount": config.fail_count,
            "closeReason": config.close_reason,
        }

    async def execute_renewal(self, db: AsyncSession) -> int:
        from app.config import settings
        from app.service.message_service import message_service

        due_configs = await self.auto_renew_repository.list_due(db)
        success_count = 0

        for config in due_configs:
            if config.fail_count >= settings.AUTO_RENEW_RETRY_MAX:
                config.status = 0
                config.close_reason = f"连续扣款失败 {settings.AUTO_RENEW_RETRY_MAX} 次，自动关闭"
                await db.flush()

                # 通知为辅助行为：续费关闭已生效，发送失败不阻断本批其余配置处理（仅记录可追踪）
                try:
                    await message_service.send(
                        db,
                        {
                            "type": "business",
                            "title": "自动续费失败通知",
                            "content": (
                                f"您的套餐自动续费已连续失败 {settings.AUTO_RENEW_RETRY_MAX} 次，"
                                "自动续费已关闭，请手动续费以保持会员权益。"
                            ),
                            "recipientIds": [config.user_id],
                            "bizModule": "auto_renew",
                            "bizId": str(config.id),
                            "priority": 2,
                        },
                    )
                except Exception:
                    logger.warning(
                        "发送自动续费失败通知失败 configId=%s", config.id, exc_info=True
                    )
                continue

            pkg = await self.package_repository.get_by_id(db, config.package_id)
            if not pkg or pkg.deleted == 1:
                continue
            if pkg.status != 1:
                continue

            now = datetime.now()
            payable_amount = int(pkg.sale_price * settings.AUTO_RENEW_DISCOUNT)
            renew_order = SysOrder(
                order_no=_gen_order_no(),
                user_id=config.user_id,
                package_id=config.package_id,
                package_name=pkg.name,
                package_type="vip",
                package_level=pkg.level_code,
                period_days=pkg.period_days,
                credit_amount=None,
                original_price=pkg.original_price,
                discount_amount=pkg.sale_price - payable_amount,
                coupon_id=None,
                coupon_amount=0,
                payable_amount=payable_amount,
                balance_amount=0,
                paid_amount=0,
                pay_method=config.pay_method,
                status=1,
                expire_time=now + timedelta(minutes=ORDER_EXPIRE_MINUTES),
                is_auto_renew=1,
            )
            await self.order_repository.create(db, renew_order)

            if config.pay_method == "balance":
                renew_success = False
                try:
                    await self.balance_account_service.freeze(
                        db, config.user_id, payable_amount
                    )
                    await self.payment_service.complete_payment(
                        db,
                        renew_order,
                        channel="balance",
                        payment_no=f"RENEW{renew_order.order_no}",
                    )
                    renew_success = True
                except Exception as e:
                    logger.error(
                        "自动续费扣款失败 configId=%s orderNo=%s: %s",
                        config.id,
                        renew_order.order_no,
                        e,
                    )
                    renew_success = False

                if renew_success:
                    config.fail_count = 0
                    config.next_renew_time = renew_order.package_expire_time or (
                        now + timedelta(days=pkg.period_days or 0)
                    )
                    config.last_renew_order_id = renew_order.id
                    success_count += 1
                else:
                    config.fail_count += 1
                    self._apply_retry_or_close(config, now, settings)
            else:
                # wechat/alipay 半自动：上一张续费订单已超时取消则计一次失败
                prev_failed = await self._is_prev_renew_cancelled(db, config)
                if prev_failed:
                    config.fail_count += 1
                    self._apply_retry_or_close(config, now, settings)
                else:
                    config.next_renew_time = now + timedelta(
                        hours=settings.AUTO_RENEW_RETRY_INTERVAL_HOURS
                    )
                config.last_renew_order_id = renew_order.id

            await db.flush()

        return success_count

    async def _is_prev_renew_cancelled(self, db: AsyncSession, config) -> bool:
        if not config.last_renew_order_id:
            return False
        prev = await self.order_repository.get_by_id(db, config.last_renew_order_id)
        return bool(prev and prev.status == 4)

    def _apply_retry_or_close(self, config, now: datetime, settings) -> None:
        if config.fail_count < settings.AUTO_RENEW_RETRY_MAX:
            config.next_renew_time = now + timedelta(
                hours=settings.AUTO_RENEW_RETRY_INTERVAL_HOURS
            )
        else:
            config.status = 0
            config.close_reason = (
                f"连续扣款失败 {settings.AUTO_RENEW_RETRY_MAX} 次，自动关闭"
            )


auto_renew_service = AutoRenewService()
