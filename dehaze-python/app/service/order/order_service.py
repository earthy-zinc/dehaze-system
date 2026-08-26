"""订单核心域：订单创建/取消/查询/统计/超时过期。

本模块同时承载订单域共享支撑（单号生成、VO 构建、状态映射、详情缓存失效），
支付/退款/自动续费子域从此处引用。
"""

import json
import logging
import random
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.redis import get_redis_client
from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback
from app.infrastructure.cache.redis_lock import (
    LockAcquireError,
    release_lock,
    try_lock_or_raise,
)
from app.models.entity.sys_order import SysOrder
from app.models.entity.sys_payment_record import SysPaymentRecord
from app.models.entity.sys_refund_record import SysRefundRecord
from app.repository.coupon_repository import coupon_repository, user_coupon_repository
from app.repository.mongo_audit_log_repository import mongo_audit_log_repository
from app.repository.order_repository import (
    ORDER_STATUS_MAP,
    ORDER_STATUS_REVERSE_MAP,
    order_repository,
)
from app.repository.package_repository import package_repository
from app.repository.payment_record_repository import payment_record_repository
from app.repository.refund_record_repository import (
    REFUND_STATUS_REVERSE_MAP,
    refund_record_repository,
)
from app.service.package_service import package_service
from app.service.payment_channel_service import payment_channel_service

logger = logging.getLogger(__name__)

ORDER_EXPIRE_MINUTES = 30
ORDER_LOCK_TTL = 5

PAY_METHODS = {"wechat", "alipay", "balance", "combined"}

ORDER_DETAIL_CACHE_TTL = 600


def _format_dt(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _gen_order_no() -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    rand = random.randint(100000, 999999)
    return f"DH{ts}{rand}"


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
        "reasonType": refund.reason_type,
        "reason": refund.reason,
        "usedDays": refund.used_days,
        "usedCredits": refund.used_credits,
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
        "packageType": order.package_type,
        "packageLevel": order.package_level,
        "creditAmount": order.credit_amount,
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


async def _invalidate_order_detail_cache(order_no: str) -> None:
    cache_key = f"order:detail:{order_no}"

    async def _del():
        redis = await get_redis_client()
        await redis.delete(cache_key)

    await redis_operation_with_fallback(_del, default=None, operation_name="order_cache_invalidate")


class OrderService:
    def __init__(
        self,
        coupon_repository=coupon_repository,
        user_coupon_repository=user_coupon_repository,
        mongo_audit_log_repository=mongo_audit_log_repository,
        order_repository=order_repository,
        package_repository=package_repository,
        package_service=package_service,
        payment_record_repository=payment_record_repository,
        refund_record_repository=refund_record_repository,
        payment_channel_service=payment_channel_service,
        balance_account_service=None,
    ):
        self.coupon_repository = coupon_repository
        self.user_coupon_repository = user_coupon_repository
        self.mongo_audit_log_repository = mongo_audit_log_repository
        self.order_repository = order_repository
        self.package_repository = package_repository
        self.package_service = package_service
        self.payment_record_repository = payment_record_repository
        self.refund_record_repository = refund_record_repository
        self.payment_channel_service = payment_channel_service
        if balance_account_service is None:
            from app.service.order.balance_account_service import balance_account_service as _b

            balance_account_service = _b
        self.balance_account_service = balance_account_service

    async def create(self, db: AsyncSession, form: dict, user_id: int) -> dict:
        package_id = form["packageId"]
        coupon_id = form.get("couponId")
        pay_method = form["payMethod"]
        balance_amount = form.get("balanceAmount") or 0

        if pay_method not in PAY_METHODS:
            raise BusinessException(ResultCode.PARAM_ERROR, "不支持的支付方式")

        lock_key = f"order:lock:{user_id}:{package_id}"
        try:
            lock_token = await try_lock_or_raise(lock_key, ORDER_LOCK_TTL, "请勿短时间内重复下单")
        except LockAcquireError as e:
            raise BusinessException(ResultCode.DUPLICATE_ORDER, str(e)) from None

        try:
            pkg = await self.package_repository.get_by_id(db, package_id)
            if not pkg or pkg.deleted == 1:
                raise BusinessException(ResultCode.PACKAGE_NOT_FOUND)
            if pkg.status != 1:
                raise BusinessException(ResultCode.PACKAGE_OFF_SHELF)

            price = await self.package_service.calculate_price(
                db, package_id, coupon_id, user_id
            )
            original_price = price["originalPrice"]
            discount_amount = price["discountAmount"]
            coupon_amount = price["couponAmount"]
            payable_amount = price["payableAmount"]

            if pay_method == "combined":
                if not (0 < balance_amount < payable_amount):
                    raise BusinessException(ResultCode.PARAM_ERROR, "组合支付余额部分金额非法")

            if coupon_id:
                locked = await self.user_coupon_repository.lock_coupon(db, coupon_id)
                if not locked:
                    raise BusinessException(ResultCode.COUPON_LOCK_FAILED)

            now = datetime.now()
            order = SysOrder(
                order_no=_gen_order_no(),
                user_id=user_id,
                package_id=package_id,
                package_name=pkg.name,
                package_type=pkg.package_type,
                package_level=pkg.level_code if pkg.package_type == "vip" else None,
                period_days=pkg.period_days if pkg.package_type == "vip" else None,
                credit_amount=pkg.credit_amount if pkg.package_type == "credit" else None,
                original_price=original_price,
                discount_amount=discount_amount,
                coupon_id=coupon_id,
                coupon_amount=coupon_amount,
                payable_amount=payable_amount,
                balance_amount=balance_amount if pay_method == "combined" else 0,
                paid_amount=0,
                pay_method=pay_method,
                status=1,
                expire_time=now + timedelta(minutes=ORDER_EXPIRE_MINUTES),
                is_auto_renew=0,
            )
            await self.order_repository.create(db, order)

            self.mongo_audit_log_repository.create_audit_async(
                operator_id=user_id,
                target_type="order",
                target_id=order.order_no,
                action="create",
                module="order",
                after_value=form if not hasattr(form, "dict") else form,
            )

            return {
                "orderNo": order.order_no,
                "payMethod": pay_method,
                "paid": False,
            }
        finally:
            await release_lock(lock_key, lock_token)

    async def cancel(self, db: AsyncSession, order_no: str, reason: str, user_id: int) -> None:
        order = await self.order_repository.get_by_order_no(db, order_no)
        if not order:
            raise BusinessException(ResultCode.ORDER_NOT_FOUND)
        if order.user_id != user_id:
            raise BusinessException(ResultCode.ORDER_NOT_FOUND)

        if order.status != 1:
            raise BusinessException(ResultCode.ORDER_STATUS_INVALID)

        if order.coupon_id:
            await self.user_coupon_repository.release_coupon(db, order.coupon_id)

        if order.pay_method in ("wechat", "alipay"):
            try:
                await self.payment_channel_service.close_order(order.pay_method, order_no)
            except Exception as e:
                logger.warning("关闭渠道订单失败 orderNo=%s: %s", order_no, e)

        order.status = 4
        order.cancel_reason = reason
        await db.flush()
        await _invalidate_order_detail_cache(order_no)

        self.mongo_audit_log_repository.create_audit_async(
            operator_id=user_id,
            target_type="order",
            target_id=order_no,
            action="cancel",
            module="order",
            after_value={"reason": reason},
        )

    async def get_detail(self, db: AsyncSession, order_no: str, user_id: int | None = None) -> dict:
        cache_key = f"order:detail:{order_no}"

        async def _get_cache():
            redis = await get_redis_client()
            data = await redis.get(cache_key)
            return data

        cached_raw = await redis_operation_with_fallback(
            _get_cache, default=None, operation_name="order_cache_get"
        )
        if cached_raw:
            try:
                cached = json.loads(cached_raw)
            except (json.JSONDecodeError, TypeError):
                cached = None
            if cached and (
                user_id is None or cached.get("userId") == user_id or cached.get("_admin") is True
            ):
                return cached

        data = await self.order_repository.get_with_user(db, order_no)
        if not data:
            raise BusinessException(ResultCode.ORDER_NOT_FOUND)
        order = data["order"]
        if user_id is not None and order.user_id != user_id:
            raise BusinessException(ResultCode.ORDER_NOT_FOUND)

        vo = _build_admin_order_vo(order, data.get("username") or "")
        vo["balanceAmount"] = order.balance_amount
        vo["expireTime"] = _format_dt(order.expire_time)
        vo["effectiveTime"] = _format_dt(order.effective_time)
        vo["cancelReason"] = order.cancel_reason
        vo["isAutoRenew"] = order.is_auto_renew

        payments = await self.payment_record_repository.list_by_order_id(db, order.id)
        if payments:
            vo["paymentRecords"] = [_payment_to_vo(p) for p in payments]

        refund = await self.refund_record_repository.get_by_order_id(db, order.id)
        if refund:
            vo["refundRecord"] = _refund_to_vo(refund, order.order_no, data.get("username") or "")

        vo["_admin"] = user_id is None

        async def _set_cache():
            redis = await get_redis_client()
            await redis.setex(
                cache_key, ORDER_DETAIL_CACHE_TTL, json.dumps(vo, ensure_ascii=False, default=str)
            )

        await redis_operation_with_fallback(
            _set_cache, default=None, operation_name="order_cache_set"
        )

        return vo

    async def list_my(self, db: AsyncSession, user_id: int, query: dict) -> dict:
        items, total = await self.order_repository.get_my_page(
            db,
            user_id,
            query["pageNum"],
            query["pageSize"],
            status=query.get("status"),
        )
        list_data = [_build_my_order_vo(o) for o in items]
        return {"list": list_data, "total": total}

    async def list_paged(self, db: AsyncSession, query: dict, current_user=None) -> dict:
        items, total = await self.order_repository.get_page(
            db,
            query["pageNum"],
            query["pageSize"],
            order_no=query.get("orderNo"),
            keywords=query.get("keywords"),
            status=query.get("status"),
            package_type=query.get("packageType"),
            pay_method=query.get("payMethod"),
            amount_min=query.get("amountMin"),
            amount_max=query.get("amountMax"),
            paid_time_start=query.get("paidTimeStart"),
            paid_time_end=query.get("paidTimeEnd"),
            current_user=current_user,
        )
        list_data = [
            _build_admin_order_vo(item["order"], item.get("username") or "") for item in items
        ]
        return {"list": list_data, "total": total}

    async def get_stats(self, db: AsyncSession, start_time: str | None, end_time: str | None) -> dict:
        base_stats = await self.order_repository.get_stats(db, start_time, end_time)
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
            pkg_dist_stmt = pkg_dist_stmt.where(
                SysOrder.create_time >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            )
        if end_time:
            pkg_dist_stmt = pkg_dist_stmt.where(
                SysOrder.create_time <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            )
        pkg_rows = (await db.execute(pkg_dist_stmt)).all()
        package_distribution = [
            {
                "packageId": row.package_id,
                "packageName": row.package_name,
                "count": row.count,
                "revenue": int(row.revenue or 0),
            }
            for row in pkg_rows
        ]

        daily_stmt = (
            select(
                func.date(SysOrder.create_time).label("date"),
                func.count().label("count"),
                func.coalesce(func.sum(SysOrder.paid_amount), 0).label("revenue"),
            )
            .where(
                SysOrder.deleted == 0,
                SysOrder.status.in_([2, 3]),
            )
            .group_by(func.date(SysOrder.create_time))
            .order_by(func.date(SysOrder.create_time).desc())
            .limit(30)
        )
        if start_time:
            daily_stmt = daily_stmt.where(
                SysOrder.create_time >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            )
        if end_time:
            daily_stmt = daily_stmt.where(
                SysOrder.create_time <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            )
        daily_rows = (await db.execute(daily_stmt)).all()
        daily_stats = [
            {
                "date": str(row.date),
                "count": row.count,
                "revenue": int(row.revenue or 0),
            }
            for row in daily_rows
        ]

        pkg_type_stmt = (
            select(
                SysOrder.package_type,
                func.count(),
                func.coalesce(func.sum(SysOrder.paid_amount), 0),
            )
            .where(SysOrder.deleted == 0, SysOrder.status.in_([2, 3]))
            .group_by(SysOrder.package_type)
        )
        if start_time:
            pkg_type_stmt = pkg_type_stmt.where(
                SysOrder.create_time >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            )
        if end_time:
            pkg_type_stmt = pkg_type_stmt.where(
                SysOrder.create_time <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            )
        pkg_type_rows = (await db.execute(pkg_type_stmt)).all()
        package_type_distribution = [
            {"packageType": row[0], "count": row[1], "revenue": int(row[2] or 0)}
            for row in pkg_type_rows
        ]

        refund_reason_stmt = (
            select(SysRefundRecord.reason_type, func.count())
            .where(SysRefundRecord.deleted == 0)
            .group_by(SysRefundRecord.reason_type)
        )
        if start_time:
            refund_reason_stmt = refund_reason_stmt.where(
                SysRefundRecord.apply_time >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            )
        if end_time:
            refund_reason_stmt = refund_reason_stmt.where(
                SysRefundRecord.apply_time <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            )
        refund_reason_rows = (await db.execute(refund_reason_stmt)).all()
        refund_reason_distribution = {
            reason_type: count for reason_type, count in refund_reason_rows
        }

        return {
            "totalOrders": total_orders,
            "totalRevenue": total_revenue,
            "totalRefund": total_refund,
            "refundRate": float(refund_rate),
            "statusDistribution": status_distribution,
            "payMethodDistribution": pay_method_distribution,
            "packageDistribution": package_distribution,
            "packageTypeDistribution": package_type_distribution,
            "refundReasonDistribution": refund_reason_distribution,
            "dailyStats": daily_stats,
        }

    async def expire_orders(self, db: AsyncSession) -> int:
        orders = await self.order_repository.list_expired_pending(db)
        count = 0
        for order in orders:
            if order.coupon_id:
                await self.user_coupon_repository.release_coupon(db, order.coupon_id)
            if order.pay_method in ("wechat", "alipay"):
                try:
                    await self.payment_channel_service.close_order(order.pay_method, order.order_no)
                except Exception as e:
                    logger.warning("超时关单失败 orderNo=%s: %s", order.order_no, e)
            if order.pay_method in ("balance", "combined"):
                frozen = (
                    order.balance_amount if order.pay_method == "combined" else order.payable_amount
                )
                if frozen > 0:
                    await self.balance_account_service.unfreeze(db, order.user_id, frozen)
            order.status = 4
            order.cancel_reason = "超时未支付，系统自动取消"
            await _invalidate_order_detail_cache(order.order_no)
            count += 1
        if count > 0:
            await db.flush()
        return count

    async def complete_expired_orders(self, db: AsyncSession) -> int:
        orders = await self.order_repository.list_completed_expiring(db)
        count = 0
        for order in orders:
            order.status = 3
            await _invalidate_order_detail_cache(order.order_no)
            count += 1
        if count > 0:
            await db.flush()
        return count


order_service = OrderService()
