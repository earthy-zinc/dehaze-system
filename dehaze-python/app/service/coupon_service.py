from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.redis import get_redis_client
from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback
from app.models.entity.sys_coupon import SysCoupon
from app.models.entity.sys_member import SysMember
from app.models.entity.sys_user import SysUser
from app.models.entity.sys_user_coupon import SysUserCoupon
from app.repository.coupon_repository import coupon_repository, user_coupon_repository


def _format_dt(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _parse_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _calc_expire_time(coupon: SysCoupon, receive_time: datetime) -> datetime | None:
    if coupon.valid_type == "fixed":
        return coupon.valid_end
    if coupon.valid_type == "relative" and coupon.valid_days:
        return receive_time + timedelta(days=coupon.valid_days)
    return None


def _coupon_to_vo(coupon: SysCoupon) -> dict:
    return {
        "id": coupon.id,
        "name": coupon.name,
        "type": coupon.type,
        "faceValue": coupon.face_value,
        "threshold": coupon.threshold,
        "validType": coupon.valid_type,
        "validStart": _format_dt(coupon.valid_start),
        "validEnd": _format_dt(coupon.valid_end),
        "validDays": coupon.valid_days,
        "totalQty": coupon.total_qty,
        "issuedQty": coupon.issued_qty,
        "usedQty": coupon.used_qty,
        "perUserLimit": coupon.per_user_limit,
        "applicableScope": coupon.applicable_scope,
        "status": coupon.status,
        "createTime": _format_dt(coupon.create_time),
    }


class CouponService:
    async def create(self, db: AsyncSession, form: dict) -> dict:
        if form["type"] == "full_reduction" and form.get("threshold") is None:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "满减券必须设置使用门槛")
        if form["validType"] == "fixed" and (not form.get("validStart") or not form.get("validEnd")):
            raise BusinessException(ResultCode.BUSINESS_ERROR, "固定有效期必须设置起止时间")
        coupon = SysCoupon(
            name=form["name"],
            type=form["type"],
            face_value=form["faceValue"],
            threshold=form.get("threshold"),
            valid_type=form["validType"],
            valid_start=_parse_dt(form["validStart"]) if form.get("validStart") else None,
            valid_end=_parse_dt(form["validEnd"]) if form.get("validEnd") else None,
            valid_days=form.get("validDays"),
            total_qty=form["totalQty"],
            per_user_limit=form["perUserLimit"],
            applicable_scope=form.get("applicableScope"),
            status=form.get("status", 1),
        )
        await coupon_repository.create(db, coupon)
        return {"id": coupon.id}

    async def update(self, db: AsyncSession, coupon_id: int, form: dict) -> None:
        coupon = await coupon_repository.get_by_id(db, coupon_id)
        if not coupon:
            raise BusinessException(ResultCode.COUPON_NOT_FOUND)
        if form["type"] == "full_reduction" and form.get("threshold") is None:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "满减券必须设置使用门槛")
        if form["validType"] == "fixed" and (not form.get("validStart") or not form.get("validEnd")):
            raise BusinessException(ResultCode.BUSINESS_ERROR, "固定有效期必须设置起止时间")
        coupon.name = form["name"]
        coupon.type = form["type"]
        coupon.face_value = form["faceValue"]
        coupon.threshold = form.get("threshold")
        coupon.valid_type = form["validType"]
        coupon.valid_start = _parse_dt(form["validStart"]) if form.get("validStart") else None
        coupon.valid_end = _parse_dt(form["validEnd"]) if form.get("validEnd") else None
        coupon.valid_days = form.get("validDays")
        coupon.total_qty = form["totalQty"]
        coupon.per_user_limit = form["perUserLimit"]
        coupon.applicable_scope = form.get("applicableScope")
        if form.get("status") is not None:
            coupon.status = form["status"]
        await db.flush()

    async def delete_by_ids(self, db: AsyncSession, ids: list[int]) -> None:
        for coupon_id in ids:
            coupon = await coupon_repository.get_by_id(db, coupon_id)
            if not coupon:
                raise BusinessException(ResultCode.COUPON_NOT_FOUND)
        used_count = await user_coupon_repository.count_used_by_coupon_ids(db, ids)
        if used_count > 0:
            raise BusinessException(ResultCode.DATA_BIND_EXISTS, "优惠券已发放使用，无法删除")
        await user_coupon_repository.soft_delete_unused_by_coupon_ids(db, ids)
        await coupon_repository.soft_delete_by_ids(db, ids)

    async def get_page(self, db: AsyncSession, query: dict) -> dict:
        items, total = await coupon_repository.get_page(
            db,
            query["pageNum"],
            query["pageSize"],
            name=query.get("name"),
            type=query.get("type"),
            status=query.get("status"),
        )
        list_data = [_coupon_to_vo(c) for c in items]
        return {"list": list_data, "total": total}

    async def batch_distribute(self, db: AsyncSession, form: dict) -> dict:
        coupon_id = form["couponId"]
        coupon = await coupon_repository.get_by_id(db, coupon_id)
        if not coupon:
            raise BusinessException(ResultCode.COUPON_NOT_FOUND)
        if coupon.status != 1:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "优惠券已禁用")

        target_scope = form["targetScope"]
        user_ids: list[int] = []

        if target_scope == "users":
            user_ids = form.get("userIds") or []
        elif target_scope == "level":
            level_codes = form.get("levelCodes") or []
            if level_codes:
                stmt = select(SysMember.user_id).where(
                    SysMember.deleted == 0,
                    SysMember.level_code.in_(level_codes),
                )
                result = await db.execute(stmt)
                user_ids = list(result.scalars().all())
        elif target_scope == "all":
            stmt = select(SysUser.id).where(SysUser.deleted == 0, SysUser.status == 1)
            result = await db.execute(stmt)
            user_ids = list(result.scalars().all())

        success_count = 0
        fail_count = 0
        now = datetime.now()
        expire_time = _calc_expire_time(coupon, now)

        for uid in user_ids:
            try:
                existing_count = await user_coupon_repository.count_by_user_and_coupon(
                    db, uid, coupon_id
                )
                if existing_count >= coupon.per_user_limit:
                    fail_count += 1
                    continue

                if coupon.total_qty != -1 and coupon.issued_qty + success_count >= coupon.total_qty:
                    fail_count += 1
                    continue

                user_coupon = SysUserCoupon(
                    user_id=uid,
                    coupon_id=coupon_id,
                    status=1,
                    receive_time=now,
                    expire_time=expire_time,
                )
                await user_coupon_repository.create(db, user_coupon)
                success_count += 1
            except Exception:
                fail_count += 1

        if success_count > 0:
            await coupon_repository.increment_issued_qty_with_limit(db, coupon_id, success_count)

        return {"successCount": success_count, "failCount": fail_count}

    async def receive(self, db: AsyncSession, coupon_id: int, user_id: int) -> dict:
        coupon = await coupon_repository.get_by_id(db, coupon_id)
        if not coupon:
            raise BusinessException(ResultCode.COUPON_NOT_FOUND)
        if coupon.status != 1:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "优惠券已禁用")

        if coupon.type == "trial":
            trial_count = await user_coupon_repository.count_by_user_and_coupon(
                db, user_id, coupon_id
            )
            if trial_count > 0:
                raise BusinessException(ResultCode.BUSINESS_ERROR, "体验券每人限领 1 次")

        if coupon.total_qty != -1 and coupon.issued_qty >= coupon.total_qty:
            raise BusinessException(ResultCode.COUPON_STOCK_EMPTY)

        existing_count = await user_coupon_repository.count_by_user_and_coupon(
            db, user_id, coupon_id
        )
        if existing_count >= coupon.per_user_limit:
            raise BusinessException(ResultCode.COUPON_LIMIT_EXCEEDED)

        rate_key = f"coupon:receive:rate:{user_id}"

        async def _check_rate():
            redis = await get_redis_client()
            count = await redis.incr(rate_key)
            if count == 1:
                await redis.expire(rate_key, settings.COUPON_RECEIVE_RATE_WINDOW)
            return count

        count = await redis_operation_with_fallback(
            operation=_check_rate,
            default=0,
            operation_name="coupon_receive_rate_limit",
        )
        if count and count > settings.COUPON_RECEIVE_RATE_LIMIT:
            raise BusinessException(ResultCode.RATE_LIMIT)

        success = await coupon_repository.increment_issued_qty_with_limit(db, coupon_id)
        if not success:
            raise BusinessException(ResultCode.COUPON_STOCK_EMPTY)

        now = datetime.now()
        expire_time = _calc_expire_time(coupon, now)
        user_coupon = SysUserCoupon(
            user_id=user_id,
            coupon_id=coupon_id,
            status=1,
            receive_time=now,
            expire_time=expire_time,
        )
        await user_coupon_repository.create(db, user_coupon)
        return {"userCouponId": user_coupon.id}

    async def list_my(self, db: AsyncSession, user_id: int, status: int | None) -> list[dict]:
        user_coupons = await user_coupon_repository.list_by_user(db, user_id, status)
        if not user_coupons:
            return []
        coupon_ids = list({uc.coupon_id for uc in user_coupons})
        coupons = await coupon_repository.get_by_ids(db, coupon_ids, with_deleted=True)
        coupon_map = {c.id: c for c in coupons}

        result = []
        for uc in user_coupons:
            coupon = coupon_map.get(uc.coupon_id)
            if not coupon:
                continue
            result.append(
                {
                    "id": uc.id,
                    "couponId": uc.coupon_id,
                    "couponName": coupon.name,
                    "type": coupon.type,
                    "faceValue": coupon.face_value,
                    "threshold": coupon.threshold,
                    "status": uc.status,
                    "receiveTime": _format_dt(uc.receive_time),
                    "expireTime": _format_dt(uc.expire_time),
                    "usedTime": _format_dt(uc.used_time),
                    "usedOrderId": uc.used_order_id,
                    "applicableScope": coupon.applicable_scope,
                }
            )
        return result

    async def expire_user_coupons(self, db: AsyncSession) -> int:
        return await user_coupon_repository.expire_coupons(db)


coupon_service = CouponService()
