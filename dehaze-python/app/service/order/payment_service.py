"""支付域：发起支付、渠道回调处理、支付完成链路（流水/订单状态/券核销/权益激活）。"""

import json
import logging
import random
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_payment_record import SysPaymentRecord
from app.repository.coupon_repository import coupon_repository, user_coupon_repository
from app.repository.order_repository import order_repository
from app.repository.package_repository import package_repository
from app.repository.payment_record_repository import payment_record_repository
from app.service.order.order_service import (
    PAY_METHODS,
    _invalidate_order_detail_cache,
)
from app.service.payment_channel_service import payment_channel_service

logger = logging.getLogger(__name__)

PAYMENT_LOCK_TTL = 10


def _gen_payment_no(channel: str) -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    rand = random.randint(100000, 999999)
    return f"PAY{channel.upper()}{ts}{rand}"


class PaymentService:
    def __init__(
        self,
        order_repository=order_repository,
        package_repository=package_repository,
        payment_record_repository=payment_record_repository,
        payment_channel_service=payment_channel_service,
        coupon_repository=coupon_repository,
        user_coupon_repository=user_coupon_repository,
        balance_account_service=None,
        member_service=None,
        ai_balance_service=None,
    ):
        self.order_repository = order_repository
        self.package_repository = package_repository
        self.payment_record_repository = payment_record_repository
        self.payment_channel_service = payment_channel_service
        self.coupon_repository = coupon_repository
        self.user_coupon_repository = user_coupon_repository
        if balance_account_service is None:
            from app.service.order.balance_account_service import (
                balance_account_service as _b,
            )

            balance_account_service = _b
        self.balance_account_service = balance_account_service
        if member_service is None:
            from app.service.member.member_service import member_service as _m

            member_service = _m
        self.member_service = member_service
        if ai_balance_service is None:
            from app.service.billing.balance_service import balance_service as _ab

            ai_balance_service = _ab
        self.ai_balance_service = ai_balance_service

    async def _prewrite_payment_record(
        self, db: AsyncSession, order, channel: str, amount: int
    ) -> None:
        """支付阶段预写"处理中"支付流水（status=1），回调成功时原地更新为成功。

        同订单已有处理中流水时复用（重复发起支付不重复预写）。
        """
        payment = await self.payment_record_repository.get_pending_by_order_id(db, order.id)
        if payment:
            payment.channel = channel
            payment.amount = amount
            await db.flush()
        else:
            payment = SysPaymentRecord(
                order_id=order.id,
                user_id=order.user_id,
                payment_no=_gen_payment_no(channel),
                channel=channel,
                amount=amount,
                status=1,
            )
            await self.payment_record_repository.create(db, payment)

    async def pay(self, db: AsyncSession, order_no: str, form: dict, user_id: int) -> dict:
        order = await self.order_repository.get_by_order_no(db, order_no)
        if not order:
            raise BusinessException(ResultCode.ORDER_NOT_FOUND)
        if order.user_id != user_id:
            raise BusinessException(ResultCode.ORDER_NOT_FOUND)

        pay_method = form["payMethod"]
        if pay_method not in PAY_METHODS:
            raise BusinessException(ResultCode.PARAM_ERROR, "不支持的支付方式")

        # 状态校验补全：已支付/已完成、已取消/退款中/已退款、超时
        if order.status in (2, 3):
            raise BusinessException(ResultCode.ORDER_ALREADY_PAID)
        if order.status in (4, 5, 6):
            raise BusinessException(ResultCode.ORDER_STATUS_INVALID)
        if order.expire_time and order.expire_time < datetime.now():
            raise BusinessException(ResultCode.ORDER_EXPIRED)

        if pay_method == "balance":
            await self.balance_account_service.freeze(
                db, user_id, order.payable_amount
            )
            order.pay_method = "balance"
            await self.complete_payment(
                db, order, channel="balance", payment_no=_gen_payment_no("balance")
            )
            return {"orderNo": order.order_no, "payMethod": "balance", "paid": True}

        if pay_method == "combined":
            balance_amount = order.balance_amount or form.get("balanceAmount") or 0
            if not (0 < balance_amount < order.payable_amount):
                raise BusinessException(ResultCode.PARAM_ERROR, "组合支付余额部分金额非法")
            await self.balance_account_service.freeze(db, user_id, balance_amount)
            order.balance_amount = balance_amount
            order.pay_method = "combined"
            await db.flush()
            third_party_amount = order.payable_amount - balance_amount
            pay_result = await self.payment_channel_service.unified_order(
                pay_method, order_no, third_party_amount, order.package_name
            )
            await self._prewrite_payment_record(
                db, order, channel=pay_method, amount=third_party_amount
            )
            return {
                "orderNo": order.order_no,
                "payMethod": pay_method,
                "payUrl": pay_result.pay_url,
                "qrCode": pay_result.qr_code,
                "paid": False,
            }

        # wechat/alipay：渠道统一下单
        order.pay_method = pay_method
        await db.flush()
        pay_result = await self.payment_channel_service.unified_order(
            pay_method, order_no, order.payable_amount, order.package_name
        )
        await self._prewrite_payment_record(
            db, order, channel=pay_method, amount=order.payable_amount
        )
        return {
            "orderNo": order.order_no,
            "payMethod": pay_method,
            "payUrl": pay_result.pay_url,
            "qrCode": pay_result.qr_code,
            "paid": False,
        }

    async def complete_payment(
        self,
        db: AsyncSession,
        order,
        *,
        channel: str,
        payment_no: str,
        callback_content: str | None = None,
    ) -> None:
        """支付完成落库与履约分流：
        订单置已支付 + 写支付流水 + 核销优惠券 + 扣减冻结余额 + 累加销量 +
        按商品类型履约（vip→会员激活；credit→积分到账并置已完成）+ 清订单缓存。
        """
        from app.repository.package_repository import package_repository as _pkg_repo

        now = datetime.now()
        # 优先更新 pay 阶段预写的处理中流水；余额支付无预写则新建
        payment = await self.payment_record_repository.get_pending_by_order_id(db, order.id)
        if payment:
            payment.payment_no = payment_no
            payment.status = 2
            payment.callback_time = now
            payment.callback_content = callback_content
            await db.flush()
        else:
            payment = SysPaymentRecord(
                order_id=order.id,
                user_id=order.user_id,
                payment_no=payment_no,
                channel=channel,
                amount=order.payable_amount,
                status=2,
                callback_time=now,
                callback_content=callback_content,
            )
            await self.payment_record_repository.create(db, payment)

        order.status = 2
        order.paid_amount = order.payable_amount
        order.paid_time = now
        order.effective_time = now
        await db.flush()

        if order.coupon_id:
            await self.user_coupon_repository.consume_coupon(db, order.coupon_id, order.id)
            uc = await self.user_coupon_repository.get_by_id(db, order.coupon_id)
            if uc:
                await self.coupon_repository.increment_used_qty(db, uc.coupon_id)

        # 履约分流
        if order.package_type == "credit":
            credit_amount = int(order.credit_amount or 0)
            if credit_amount > 0:
                await self.ai_balance_service.increase(
                    db,
                    user_id=order.user_id,
                    amount=credit_amount,
                    source="recharge",
                    related_id=order.id,
                    reason="积分卡购买到账",
                )
            order.status = 3
        else:
            pkg = await _pkg_repo.get_by_id(db, order.package_id)
            order.package_expire_time = (order.paid_time + timedelta(days=pkg.period_days)) if (
                pkg and pkg.period_days
            ) else order.paid_time

        await self.member_service.on_order_paid(db, order)

        # 扣减冻结余额（balance/combined）
        if order.pay_method in ("balance", "combined"):
            frozen = order.balance_amount if order.pay_method == "combined" else order.paid_amount
            if frozen > 0:
                await self.balance_account_service.deduct(db, order.user_id, frozen)

        # 累加商品销量
        pkg = await _pkg_repo.get_by_id(db, order.package_id)
        if pkg:
            pkg.sales_count = (pkg.sales_count or 0) + 1
            await db.flush()

        await _invalidate_order_detail_cache(order.order_no)

    async def handle_payment_callback(
        self, db: AsyncSession, channel: str, headers: dict, body: bytes
    ) -> bool:
        from app.infrastructure.cache.redis_lock import acquire_lock, release_lock

        callback = await self.payment_channel_service.verify_callback(channel, headers, body)
        if not callback.success:
            logger.warning(
                "支付回调失败 channel=%s orderNo=%s raw=%s",
                channel,
                callback.order_no,
                callback.raw,
            )
            return False

        order = await self.order_repository.get_by_order_no(db, callback.order_no)
        if not order:
            # 非商品订单：尝试按余额充值单处理（充值统一下单以充值单号为商户单号）
            from app.service.order.recharge_service import recharge_service

            return await recharge_service.handle_payment_callback(db, callback, channel)

        # 金额校验：回调金额 > 0 且与渠道应付金额一致，否则 A0538
        # （组合支付渠道仅收取第三方部分 = 应付 - 余额部分）
        expected_amount = order.payable_amount - (
            order.balance_amount if order.balance_amount > 0 else 0
        )
        if callback.amount <= 0 or callback.amount != expected_amount:
            logger.error(
                "支付回调金额不一致 orderNo=%s expected=%s actual=%s",
                order.order_no,
                expected_amount,
                callback.amount,
            )
            raise BusinessException(ResultCode.PAYMENT_AMOUNT_MISMATCH)

        # 订单级分布式锁：获取失败视为重复回调，幂等返回成功
        lock_key = f"payment:lock:{order.order_no}"
        lock_token = await acquire_lock(lock_key, PAYMENT_LOCK_TTL)
        if lock_token is None:
            logger.info("支付回调获取锁失败，幂等返回 orderNo=%s", order.order_no)
            return True

        try:
            if order.status in (2, 3):
                logger.info("支付回调幂等返回 orderNo=%s", callback.order_no)
                return True
            if order.status != 1:
                logger.warning(
                    "支付回调订单状态异常 orderNo=%s status=%s", callback.order_no, order.status
                )
                return False

            existing = await self.payment_record_repository.get_by_payment_no(
                db, callback.channel_payment_no
            )
            if existing:
                return True

            order.pay_method = "combined" if order.balance_amount > 0 else channel
            await self.complete_payment(
                db,
                order,
                channel=channel,
                payment_no=callback.channel_payment_no,
                callback_content=json.dumps(callback.raw, ensure_ascii=False),
            )
            return True
        finally:
            await release_lock(lock_key, lock_token)


payment_service = PaymentService()
