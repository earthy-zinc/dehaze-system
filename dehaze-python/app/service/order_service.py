import random
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_auto_renew import SysAutoRenew
from app.models.entity.sys_coupon import SysCoupon
from app.models.entity.sys_member import SysMember
from app.models.entity.sys_member_benefit import SysMemberBenefit
from app.models.entity.sys_order import SysOrder
from app.models.entity.sys_package import SysPackage
from app.models.entity.sys_payment_record import SysPaymentRecord
from app.models.entity.sys_refund_record import SysRefundRecord
from app.models.entity.sys_user import SysUser
from app.models.entity.sys_user_coupon import SysUserCoupon
from app.repository.auto_renew_repository import auto_renew_repository
from app.repository.coupon_repository import coupon_repository, user_coupon_repository
from app.repository.member_benefit_repository import member_benefit_repository
from app.repository.member_repository import member_repository
from app.repository.order_repository import order_repository, ORDER_STATUS_MAP, ORDER_STATUS_REVERSE_MAP
from app.repository.package_repository import package_repository
from app.repository.payment_record_repository import payment_record_repository
from app.repository.refund_record_repository import refund_record_repository, REFUND_STATUS_REVERSE_MAP

ORDER_EXPIRE_MINUTES = 30
PAYMENT_LOCK_PREFIX = "payment:lock:"
ORDER_LOCK_PREFIX = "order:lock:"
REFUND_TIME_LIMIT_DAYS = 7

PAY_METHODS = {"wechat", "alipay", "balance", "combined"}


def _format_dt(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _gen_order_no() -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    rand = random.randint(100000, 999999)
    return f"DH{ts}{rand}"


def _gen_payment_no(channel: str) -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    rand = random.randint(100000, 999999)
    return f"PAY{channel.upper()}{ts}{rand}"


def _gen_refund_no() -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    rand = random.randint(100000, 999999)
    return f"RF{ts}{rand}"


def _order_status_to_str(status: int) -> str:
    return ORDER_STATUS_REVERSE_MAP.get(status, "unknown")


def _refund_status_to_str(status: int) -> str:
    return REFUND_STATUS_REVERSE_MAP.get(status, "unknown")


def _payment_to_vo(record: SysPaymentRecord) -> dict:
    return {
        "id": record.id,
        "paymentNo": record.payment_no,
        "channel": record.channel,
        "amount": record.amount,
        "status": record.status,
        "callbackTime": _format_dt(record.callback_time),
        "createTime": _format_dt(record.create_time),
    }


def _refund_to_vo(refund: SysRefundRecord, order_no: str, username: str) -> dict:
    return {
        "id": refund.id,
        "refundNo": refund.refund_no,
        "orderId": refund.order_id,
        "orderNo": order_no,
        "userId": refund.user_id,
        "username": username,
        "refundAmount": refund.refund_amount,
        "reason": refund.reason,
        "usedQuota": refund.used_quota,
        "status": _refund_status_to_str(refund.status),
        "channel": refund.channel,
        "channelRefundNo": refund.channel_refund_no,
        "applyTime": _format_dt(refund.apply_time),
        "auditTime": _format_dt(refund.audit_time),
        "auditorId": refund.auditor_id,
        "auditRemark": refund.audit_remark,
        "refundTime": _format_dt(refund.refund_time),
        "errorMessage": refund.error_message,
    }


def _build_my_order_vo(order: SysOrder) -> dict:
    return {
        "id": order.id,
        "orderNo": order.order_no,
        "packageName": order.package_name,
        "packageLevel": order.package_level,
        "payableAmount": order.payable_amount,
        "paidAmount": order.paid_amount,
        "payMethod": order.pay_method,
        "status": _order_status_to_str(order.status),
        "createTime": _format_dt(order.create_time),
        "paidTime": _format_dt(order.paid_time),
        "packageExpireTime": _format_dt(order.package_expire_time),
    }


def _build_admin_order_vo(order: SysOrder, username: str) -> dict:
    vo = _build_my_order_vo(order)
    vo["userId"] = order.user_id
    vo["username"] = username or ""
    vo["originalPrice"] = order.original_price
    vo["discountAmount"] = order.discount_amount
    vo["couponAmount"] = order.coupon_amount
    return vo


async def _activate_member_benefits(db: AsyncSession, order: SysOrder) -> None:
    member = await member_repository.get_by_user_id(db, order.user_id)
    if not member:
        return

    pkg = await package_repository.get_by_id(db, order.package_id)
    if not pkg:
        return

    member.level_code = pkg.level_code
    member.level_source = "package"

    now = datetime.now()
    base_time = max(member.expire_time or now, now)
    member.expire_time = base_time + timedelta(days=pkg.period_days)
    order.package_expire_time = member.expire_time
    order.effective_time = now

    benefit = await member_benefit_repository.get_by_level_code(db, pkg.level_code)
    if benefit:
        overrides = pkg.benefit_overrides if isinstance(pkg.benefit_overrides, dict) else {}
        member.monthly_dehaze_quota = overrides.get("monthlyDehazeQuota", benefit.monthly_dehaze_quota)
        member.monthly_evaluate_quota = overrides.get("monthlyEvaluateQuota", benefit.monthly_evaluate_quota)

    if member.become_member_time is None:
        member.become_member_time = now

    member.total_consumption = (member.total_consumption or 0) + order.paid_amount
    await db.flush()


async def _complete_balance_payment(db: AsyncSession, order: SysOrder) -> None:
    now = datetime.now()
    payment_no = _gen_payment_no("balance")
    payment = SysPaymentRecord(
        order_id=order.id,
        user_id=order.user_id,
        payment_no=payment_no,
        channel="balance",
        amount=order.payable_amount,
        status=2,
        callback_time=now,
    )
    await payment_record_repository.create(db, payment)

    order.status = 2
    order.paid_amount = order.payable_amount
    order.paid_time = now
    order.pay_method = order.pay_method or "balance"
    await db.flush()

    if order.coupon_id:
        await user_coupon_repository.consume_coupon(db, order.coupon_id, order.id)
        await coupon_repository.increment_used_qty(db, _get_coupon_template_id(order.coupon_id))

    await _activate_member_benefits(db, order)


def _get_coupon_template_id(user_coupon_id: int) -> int:
    return user_coupon_id


class OrderService:

    @staticmethod
    async def create(db: AsyncSession, form: dict, user_id: int) -> dict:
        package_id = form["packageId"]
        coupon_id = form.get("couponId")
        pay_method = form["payMethod"]

        if pay_method not in PAY_METHODS:
            raise BusinessException(ResultCode.PARAM_ERROR, "不支持的支付方式")

        pkg = await package_repository.get_by_id(db, package_id)
        if not pkg or pkg.deleted == 1:
            raise BusinessException(ResultCode.PACKAGE_NOT_FOUND)
        if pkg.status != 1:
            raise BusinessException(ResultCode.PACKAGE_OFF_SHELF)

        original_price = pkg.sale_price
        discount_amount = 0
        coupon_amount = 0

        if coupon_id:
            user_coupon = await user_coupon_repository.get_by_id(db, coupon_id)
            if not user_coupon:
                raise BusinessException(ResultCode.COUPON_NOT_FOUND)
            if user_coupon.user_id != user_id:
                raise BusinessException(ResultCode.BUSINESS_ERROR, "优惠券不属于当前用户")
            if user_coupon.status != 1:
                raise BusinessException(ResultCode.COUPON_STATUS_INVALID)
            if user_coupon.expire_time and user_coupon.expire_time < datetime.now():
                raise BusinessException(ResultCode.COUPON_EXPIRED)

            coupon_template = await coupon_repository.get_by_id(db, user_coupon.coupon_id)
            if not coupon_template:
                raise BusinessException(ResultCode.COUPON_NOT_FOUND)
            if coupon_template.applicable_scope and package_id not in coupon_template.applicable_scope:
                raise BusinessException(ResultCode.COUPON_NOT_APPLICABLE)

            base_price = original_price - discount_amount
            if coupon_template.type == "full_reduction":
                if base_price >= (coupon_template.threshold or 0):
                    coupon_amount = coupon_template.face_value
            elif coupon_template.type == "discount":
                coupon_amount = base_price * (100 - coupon_template.face_value) // 100
            elif coupon_template.type == "no_threshold":
                coupon_amount = coupon_template.face_value
            elif coupon_template.type == "trial":
                coupon_amount = base_price

            locked = await user_coupon_repository.lock_coupon(db, coupon_id)
            if not locked:
                raise BusinessException(ResultCode.COUPON_LOCK_FAILED)

        payable_amount = max(0, original_price - discount_amount - coupon_amount)
        order_no = _gen_order_no()
        now = datetime.now()

        order = SysOrder(
            order_no=order_no,
            user_id=user_id,
            package_id=package_id,
            package_name=pkg.name,
            package_level=pkg.level_code,
            period_days=pkg.period_days,
            original_price=original_price,
            discount_amount=discount_amount,
            coupon_id=coupon_id,
            coupon_amount=coupon_amount,
            payable_amount=payable_amount,
            paid_amount=0,
            pay_method=pay_method,
            status=1,
            expire_time=now + timedelta(minutes=ORDER_EXPIRE_MINUTES),
            is_auto_renew=0,
        )
        await order_repository.create(db, order)

        if pay_method == "balance":
            await _complete_balance_payment(db, order)
            return {
                "orderNo": order.order_no,
                "payMethod": pay_method,
                "paid": True,
            }

        pay_url = f"https://mock-pay.example.com/{order_no}"
        return {
            "orderNo": order.order_no,
            "payMethod": pay_method,
            "payUrl": pay_url,
            "qrCode": pay_url,
            "paid": False,
        }

    @staticmethod
    async def pay(db: AsyncSession, order_no: str, form: dict, user_id: int) -> dict:
        order = await order_repository.get_by_order_no(db, order_no)
        if not order:
            raise BusinessException(ResultCode.ORDER_NOT_FOUND)
        if order.user_id != user_id:
            raise BusinessException(ResultCode.ORDER_NOT_FOUND)

        pay_method = form["payMethod"]
        if pay_method not in PAY_METHODS:
            raise BusinessException(ResultCode.PARAM_ERROR, "不支持的支付方式")

        if order.status == 2 or order.status == 3:
            return {
                "orderNo": order.order_no,
                "payMethod": order.pay_method or pay_method,
                "paid": True,
            }

        if order.status != 1:
            raise BusinessException(ResultCode.ORDER_STATUS_INVALID)

        if pay_method == "balance":
            order.pay_method = "balance"
            await _complete_balance_payment(db, order)
            return {
                "orderNo": order.order_no,
                "payMethod": "balance",
                "paid": True,
            }

        pay_url = f"https://mock-pay.example.com/{order_no}"
        return {
            "orderNo": order.order_no,
            "payMethod": pay_method,
            "payUrl": pay_url,
            "qrCode": pay_url,
            "paid": False,
        }

    @staticmethod
    async def cancel(db: AsyncSession, order_no: str, reason: str, user_id: int) -> None:
        order = await order_repository.get_by_order_no(db, order_no)
        if not order:
            raise BusinessException(ResultCode.ORDER_NOT_FOUND)
        if order.user_id != user_id:
            raise BusinessException(ResultCode.ORDER_NOT_FOUND)

        if order.status not in (1, 2):
            raise BusinessException(ResultCode.ORDER_STATUS_INVALID)

        if order.coupon_id:
            await user_coupon_repository.release_coupon(db, order.coupon_id)

        order.status = 4
        order.cancel_reason = reason
        await db.flush()

    @staticmethod
    async def get_detail(db: AsyncSession, order_no: str, user_id: Optional[int] = None) -> dict:
        data = await order_repository.get_with_user(db, order_no)
        if not data:
            raise BusinessException(ResultCode.ORDER_NOT_FOUND)
        order = data["order"]
        if user_id is not None and order.user_id != user_id:
            raise BusinessException(ResultCode.ORDER_NOT_FOUND)

        vo = _build_admin_order_vo(order, data.get("username") or "")
        vo["expireTime"] = _format_dt(order.expire_time)
        vo["effectiveTime"] = _format_dt(order.effective_time)
        vo["cancelReason"] = order.cancel_reason
        vo["isAutoRenew"] = order.is_auto_renew

        payments = await payment_record_repository.list_by_order_id(db, order.id)
        if payments:
            vo["paymentRecords"] = [_payment_to_vo(p) for p in payments]

        refund = await refund_record_repository.get_by_order_id(db, order.id)
        if refund:
            vo["refundRecord"] = _refund_to_vo(refund, order.order_no, data.get("username") or "")

        return vo

    @staticmethod
    async def list_my(db: AsyncSession, user_id: int, query: dict) -> dict:
        items, total = await order_repository.get_my_page(
            db,
            user_id,
            query["pageNum"],
            query["pageSize"],
            status=query.get("status"),
        )
        list_data = [_build_my_order_vo(o) for o in items]
        return {"list": list_data, "total": total}

    @staticmethod
    async def list_paged(db: AsyncSession, query: dict) -> dict:
        items, total = await order_repository.get_page(
            db,
            query["pageNum"],
            query["pageSize"],
            order_no=query.get("orderNo"),
            keywords=query.get("keywords"),
            status=query.get("status"),
            pay_method=query.get("payMethod"),
            amount_min=query.get("amountMin"),
            amount_max=query.get("amountMax"),
            paid_time_start=query.get("paidTimeStart"),
            paid_time_end=query.get("paidTimeEnd"),
        )
        list_data = [_build_admin_order_vo(item["order"], item.get("username") or "") for item in items]
        return {"list": list_data, "total": total}

    @staticmethod
    async def apply_refund(db: AsyncSession, order_no: str, form: dict, user_id: int) -> None:
        order = await order_repository.get_by_order_no(db, order_no)
        if not order:
            raise BusinessException(ResultCode.ORDER_NOT_FOUND)
        if order.user_id != user_id:
            raise BusinessException(ResultCode.ORDER_NOT_FOUND)

        if order.status != 2:
            raise BusinessException(ResultCode.ORDER_STATUS_INVALID)

        if order.paid_time and order.paid_time < datetime.now() - timedelta(days=REFUND_TIME_LIMIT_DAYS):
            raise BusinessException(ResultCode.REFUND_TIME_EXCEEDED)

        existing = await refund_record_repository.get_by_order_id(db, order.id)
        if existing:
            raise BusinessException(ResultCode.REFUND_ALREADY_EXISTS)

        reason = form["reason"]
        if form.get("customReason"):
            reason = f"{reason}:{form['customReason']}"

        refund = SysRefundRecord(
            refund_no=_gen_refund_no(),
            order_id=order.id,
            user_id=user_id,
            refund_amount=order.paid_amount,
            reason=reason,
            used_quota=0,
            status=1,
            apply_time=datetime.now(),
        )
        await refund_record_repository.create(db, refund)

        order.status = 5
        await db.flush()

    @staticmethod
    async def approve_refund(db: AsyncSession, refund_id: int, form: dict, auditor_id: int) -> None:
        refund = await refund_record_repository.get_by_id(db, refund_id)
        if not refund:
            raise BusinessException(ResultCode.REFUND_NOT_FOUND)

        if refund.status != 1:
            raise BusinessException(ResultCode.ORDER_STATUS_INVALID)

        order = await order_repository.get_by_id(db, refund.order_id)
        if not order:
            raise BusinessException(ResultCode.ORDER_NOT_FOUND)

        now = datetime.now()
        refund.status = 2
        refund.audit_time = now
        refund.auditor_id = auditor_id
        refund.audit_remark = form.get("remark", "")
        refund.refund_time = now
        refund.channel = order.pay_method
        await db.flush()

        order.status = 6
        await db.flush()

    @staticmethod
    async def reject_refund(db: AsyncSession, refund_id: int, form: dict, auditor_id: int) -> None:
        refund = await refund_record_repository.get_by_id(db, refund_id)
        if not refund:
            raise BusinessException(ResultCode.REFUND_NOT_FOUND)

        if refund.status != 1:
            raise BusinessException(ResultCode.ORDER_STATUS_INVALID)

        order = await order_repository.get_by_id(db, refund.order_id)
        if not order:
            raise BusinessException(ResultCode.ORDER_NOT_FOUND)

        now = datetime.now()
        refund.status = 3
        refund.audit_time = now
        refund.auditor_id = auditor_id
        refund.audit_remark = form.get("remark", "")
        await db.flush()

        order.status = 2
        await db.flush()

    @staticmethod
    async def list_refunds(db: AsyncSession, query: dict) -> dict:
        items, total = await refund_record_repository.get_page(
            db,
            query["pageNum"],
            query["pageSize"],
            order_no=query.get("orderNo"),
            keywords=query.get("keywords"),
            status=query.get("status"),
            apply_time_start=query.get("applyTimeStart"),
            apply_time_end=query.get("applyTimeEnd"),
        )
        list_data = [
            _refund_to_vo(item["refund"], item["order_no"], item.get("username") or "")
            for item in items
        ]
        return {"list": list_data, "total": total}

    @staticmethod
    async def get_stats(db: AsyncSession, start_time: Optional[str], end_time: Optional[str]) -> dict:
        base_stats = await order_repository.get_stats(db, start_time, end_time)
        total_orders = base_stats["total_orders"]
        total_revenue = base_stats["total_revenue"]
        total_refund = base_stats["total_refund"]
        refund_rate = (total_refund / total_revenue) if total_revenue > 0 else 0

        status_distribution = {s: 0 for s in ORDER_STATUS_MAP.keys()}
        status_distribution.update(base_stats["status_distribution"])

        pay_method_distribution = {m: 0 for m in PAY_METHODS}
        pay_method_distribution.update(base_stats["pay_method_distribution"])

        pkg_dist_stmt = (
            select(
                SysOrder.package_id,
                SysOrder.package_name,
                func.count().label("count"),
                func.coalesce(func.sum(SysOrder.paid_amount), 0).label("revenue"),
            )
            .where(SysOrder.deleted == 0, SysOrder.status.in_([2, 3]))
            .group_by(SysOrder.package_id, SysOrder.package_name)
        )
        if start_time:
            pkg_dist_stmt = pkg_dist_stmt.where(SysOrder.create_time >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"))
        if end_time:
            pkg_dist_stmt = pkg_dist_stmt.where(SysOrder.create_time <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S"))
        pkg_rows = (await db.execute(pkg_dist_stmt)).all()
        package_distribution = [
            {
                "packageId": row.package_id,
                "packageName": row.package_name,
                "count": row.count,
                "revenue": row.revenue,
            }
            for row in pkg_rows
        ]

        daily_stmt = select(
            func.date(SysOrder.create_time).label("date"),
            func.count().label("count"),
            func.coalesce(func.sum(SysOrder.paid_amount), 0).label("revenue"),
        ).where(
            SysOrder.deleted == 0,
            SysOrder.status.in_([2, 3]),
        ).group_by(func.date(SysOrder.create_time)).order_by(func.date(SysOrder.create_time).desc()).limit(30)
        if start_time:
            daily_stmt = daily_stmt.where(SysOrder.create_time >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"))
        if end_time:
            daily_stmt = daily_stmt.where(SysOrder.create_time <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S"))
        daily_rows = (await db.execute(daily_stmt)).all()
        daily_stats = [
            {
                "date": str(row.date),
                "count": row.count,
                "revenue": row.revenue,
            }
            for row in daily_rows
        ]

        return {
            "totalOrders": total_orders,
            "totalRevenue": total_revenue,
            "totalRefund": total_refund,
            "refundRate": refund_rate,
            "statusDistribution": status_distribution,
            "payMethodDistribution": pay_method_distribution,
            "packageDistribution": package_distribution,
            "dailyStats": daily_stats,
        }

    @staticmethod
    async def update_auto_renew_config(db: AsyncSession, form: dict, user_id: int) -> None:
        package_id = form["packageId"]
        pkg = await package_repository.get_by_id(db, package_id)
        if not pkg:
            raise BusinessException(ResultCode.PACKAGE_NOT_FOUND)

        pay_method = form["payMethod"]
        enabled = form["enabled"]

        config = await auto_renew_repository.get_by_user_and_package(db, user_id, package_id)
        if config:
            config.pay_method = pay_method
            if enabled:
                config.status = 1
                config.fail_count = 0
                config.close_reason = None
                last_order_stmt = select(SysOrder).where(
                    SysOrder.user_id == user_id,
                    SysOrder.package_id == package_id,
                    SysOrder.status.in_([2, 3]),
                    SysOrder.deleted == 0,
                ).order_by(SysOrder.id.desc()).limit(1)
                last_order = (await db.execute(last_order_stmt)).scalar_one_or_none()
                if last_order and last_order.package_expire_time:
                    config.next_renew_time = last_order.package_expire_time
            else:
                config.status = 0
                config.close_reason = "用户关闭"
            await db.flush()
        else:
            config = SysAutoRenew(
                user_id=user_id,
                package_id=package_id,
                pay_method=pay_method,
                status=1 if enabled else 0,
                close_reason=None if enabled else "用户关闭",
            )
            if enabled:
                last_order_stmt = select(SysOrder).where(
                    SysOrder.user_id == user_id,
                    SysOrder.package_id == package_id,
                    SysOrder.status.in_([2, 3]),
                    SysOrder.deleted == 0,
                ).order_by(SysOrder.id.desc()).limit(1)
                last_order = (await db.execute(last_order_stmt)).scalar_one_or_none()
                if last_order and last_order.package_expire_time:
                    config.next_renew_time = last_order.package_expire_time
            await auto_renew_repository.create(db, config)

    @staticmethod
    async def get_auto_renew_config(db: AsyncSession, package_id: int, user_id: int) -> dict:
        pkg = await package_repository.get_by_id(db, package_id)
        if not pkg:
            raise BusinessException(ResultCode.PACKAGE_NOT_FOUND)

        config = await auto_renew_repository.get_by_user_and_package(db, user_id, package_id)
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

    @staticmethod
    async def expire_orders(db: AsyncSession) -> int:
        orders = await order_repository.list_expired_pending(db)
        count = 0
        for order in orders:
            if order.coupon_id:
                await user_coupon_repository.release_coupon(db, order.coupon_id)
            order.status = 4
            order.cancel_reason = "超时未支付，系统自动取消"
            count += 1
        if count > 0:
            await db.flush()
        return count

    @staticmethod
    async def complete_expired_orders(db: AsyncSession) -> int:
        orders = await order_repository.list_completed_expiring(db)
        count = 0
        for order in orders:
            order.status = 3
            count += 1
        if count > 0:
            await db.flush()
        return count
