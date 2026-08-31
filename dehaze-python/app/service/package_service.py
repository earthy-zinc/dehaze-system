import json
import logging
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.redis import get_redis_client
from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback
from app.models.entity.sys_coupon import SysCoupon
from app.models.entity.sys_member_benefit import SysMemberBenefit
from app.models.entity.sys_order import SysOrder
from app.models.entity.sys_package import SysPackage
from app.models.entity.sys_promotion import SysPromotion
from app.repository.coupon_repository import user_coupon_repository
from app.repository.member_benefit_repository import member_benefit_repository
from app.repository.package_repository import package_repository
from app.repository.promotion_repository import promotion_repository

logger = logging.getLogger(__name__)

PACKAGE_ONSALE_CACHE_TTL = 300
PACKAGE_DETAIL_CACHE_TTL = 600

BENEFIT_FIELDS = [
    "monthlyDehazeQuota",
    "monthlyEvaluateQuota",
    "historyRetention",
    "batchLimit",
    "priority",
    "advancedParams",
    "hdExport",
    "reportExport",
    "batchDownload",
]

PERIOD_NAMES = {
    "monthly": "月卡",
    "quarterly": "季卡",
    "yearly": "年卡",
}

VALID_PERIODS = {"monthly", "quarterly", "yearly"}


VALID_PACKAGE_TYPES = {"vip", "credit"}

PACKAGE_TYPE_NAMES = {
    "vip": "会员卡",
    "credit": "积分卡",
}


def _validate_package_form(form: dict, package_type: str) -> None:
    if package_type not in VALID_PACKAGE_TYPES:
        raise BusinessException(ResultCode.PARAM_ERROR, "商品类型非法")
    if form.get("salePrice", 0) > form.get("originalPrice", 0):
        raise BusinessException(ResultCode.PARAM_ERROR, "促销价不能高于原价")
    if package_type == "vip":
        if not form.get("levelCode"):
            raise BusinessException(ResultCode.PARAM_ERROR, "会员卡必须设置等级/周期/有效期")
        if not form.get("period"):
            raise BusinessException(ResultCode.PARAM_ERROR, "会员卡必须设置等级/周期/有效期")
        if form.get("periodDays") is None:
            raise BusinessException(ResultCode.PARAM_ERROR, "会员卡必须设置等级/周期/有效期")
        if form.get("period") not in VALID_PERIODS:
            raise BusinessException(ResultCode.PARAM_ERROR, "计费周期非法")
    elif package_type == "credit":
        if not form.get("creditAmount") or form["creditAmount"] <= 0:
            raise BusinessException(ResultCode.PARAM_ERROR, "积分卡可得积分必须大于0")


def _format_dt(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _get_effective_benefits(benefit: SysMemberBenefit | None, overrides: dict | None) -> dict:
    if not benefit:
        return {}
    base = {
        "monthlyDehazeQuota": benefit.monthly_dehaze_quota,
        "monthlyEvaluateQuota": benefit.monthly_evaluate_quota,
        "historyRetention": benefit.history_retention,
        "batchLimit": benefit.batch_limit,
        "priority": benefit.priority,
        "advancedParams": benefit.advanced_params,
        "hdExport": benefit.hd_export,
        "reportExport": benefit.report_export,
        "batchDownload": benefit.batch_download,
    }
    if overrides:
        for key in BENEFIT_FIELDS:
            if key in overrides and overrides[key] is not None:
                base[key] = overrides[key]
    return base


def _promotion_to_vo(promotion: SysPromotion) -> dict:
    return {
        "id": promotion.id,
        "name": promotion.name,
        "type": promotion.type,
        "description": promotion.description,
        "startTime": _format_dt(promotion.start_time),
        "endTime": _format_dt(promotion.end_time),
        "activityRules": promotion.activity_rules,
        "newUserOnly": promotion.new_user_only,
        "status": promotion.status,
    }


def _calc_daily_price(sale_price: int, period_days: int | None) -> int:
    if not period_days or period_days <= 0:
        return 0
    return (2 * sale_price + period_days) // (2 * period_days)


async def _invalidate_package_cache(package_id: int | None = None) -> None:
    keys = ["package:onsale:all", "package:onsale:vip", "package:onsale:credit"]
    if package_id is not None:
        keys.append(f"package:detail:{package_id}")

    async def _del():
        redis = await get_redis_client()
        await redis.delete(*keys)

    await redis_operation_with_fallback(
        _del, default=None, operation_name="package_cache_invalidate"
    )


def _build_package_detail(db_pkg: SysPackage, benefit: SysMemberBenefit | None) -> dict:
    overrides = db_pkg.benefit_overrides if isinstance(db_pkg.benefit_overrides, dict) else None
    benefits = _get_effective_benefits(benefit, overrides)
    return {
        "id": db_pkg.id,
        "name": db_pkg.name,
        "packageType": db_pkg.package_type,
        "levelCode": db_pkg.level_code,
        "levelName": benefit.level_name if benefit else "",
        "period": db_pkg.period,
        "periodDays": db_pkg.period_days,
        "originalPrice": db_pkg.original_price,
        "salePrice": db_pkg.sale_price,
        "dailyPrice": _calc_daily_price(db_pkg.sale_price, db_pkg.period_days),
        "creditAmount": db_pkg.credit_amount,
        "creditUnitPrice": _calc_credit_unit_price(db_pkg.sale_price, db_pkg.credit_amount),
        "description": db_pkg.description,
        "benefits": benefits,
        "salesCount": db_pkg.sales_count,
    }


def _calc_credit_unit_price(sale_price: int, credit_amount: int | None) -> int:
    if not credit_amount or credit_amount <= 0:
        return 0
    return sale_price // credit_amount


def _build_package_list_vo(db_pkg: SysPackage, level_name: str = "") -> dict:
    return {
        "id": db_pkg.id,
        "name": db_pkg.name,
        "packageType": db_pkg.package_type,
        "levelCode": db_pkg.level_code,
        "levelName": level_name,
        "period": db_pkg.period,
        "periodDays": db_pkg.period_days,
        "originalPrice": db_pkg.original_price,
        "salePrice": db_pkg.sale_price,
        "dailyPrice": _calc_daily_price(db_pkg.sale_price, db_pkg.period_days),
        "creditAmount": db_pkg.credit_amount,
        "creditUnitPrice": _calc_credit_unit_price(db_pkg.sale_price, db_pkg.credit_amount),
        "salesCount": db_pkg.sales_count,
    }


class PackageService:
    async def list_on_sale(self, db: AsyncSession, package_type: str | None = None) -> list[dict]:
        cache_key = f"package:onsale:{package_type or 'all'}"

        async def _get_cache():
            redis = await get_redis_client()
            return await redis.get(cache_key)

        cached_raw = await redis_operation_with_fallback(
            _get_cache, default=None, operation_name="package_onsale_cache_get"
        )
        if cached_raw:
            try:
                return json.loads(cached_raw)
            except (json.JSONDecodeError, TypeError):
                pass

        packages = await package_repository.list_on_sale(db, package_type)
        if not packages:
            return []
        benefits = await member_benefit_repository.list_all(db)
        benefit_map = {b.level_code: b for b in benefits}
        result = []
        for pkg in packages:
            benefit = benefit_map.get(pkg.level_code)
            result.append(_build_package_detail(pkg, benefit))

        async def _set_cache():
            redis = await get_redis_client()
            await redis.setex(
                cache_key,
                PACKAGE_ONSALE_CACHE_TTL,
                json.dumps(result, ensure_ascii=False, default=str),
            )

        await redis_operation_with_fallback(
            _set_cache, default=None, operation_name="package_onsale_cache_set"
        )

        return result

    async def get_detail(self, db: AsyncSession, package_id: int) -> dict:
        cache_key = f"package:detail:{package_id}"

        async def _get_cache():
            redis = await get_redis_client()
            return await redis.get(cache_key)

        cached_raw = await redis_operation_with_fallback(
            _get_cache, default=None, operation_name="package_detail_cache_get"
        )
        if cached_raw:
            try:
                return json.loads(cached_raw)
            except (json.JSONDecodeError, TypeError):
                pass

        pkg = await package_repository.get_by_id(db, package_id)
        if not pkg:
            raise BusinessException(ResultCode.PACKAGE_NOT_FOUND)
        # 用户端详情仅提供在售套餐（T-PM-004）；后台编辑走 /form 端点不受影响。
        # 缓存路径无此问题：上下架/修改/删除均会失效 package:detail 缓存。
        if pkg.status == 0:
            raise BusinessException(ResultCode.PACKAGE_OFF_SHELF)
        benefit = await member_benefit_repository.get_by_level_code(db, pkg.level_code)
        detail = _build_package_detail(pkg, benefit)

        active_promos = await promotion_repository.list_active_by_package_id(db, package_id)
        detail["activePromotions"] = (
            [_promotion_to_vo(item["promotion"]) for item in active_promos] if active_promos else []
        )

        async def _set_cache():
            redis = await get_redis_client()
            await redis.setex(
                cache_key,
                PACKAGE_DETAIL_CACHE_TTL,
                json.dumps(detail, ensure_ascii=False, default=str),
            )

        await redis_operation_with_fallback(
            _set_cache, default=None, operation_name="package_detail_cache_set"
        )

        return detail

    async def get_page(self, db: AsyncSession, query: dict) -> dict:
        items, total = await package_repository.get_page(
            db,
            query["pageNum"],
            query["pageSize"],
            name=query.get("name"),
            package_type=query.get("packageType"),
            level_code=query.get("levelCode"),
            period=query.get("period"),
            status=query.get("status"),
            start_time=query.get("startTime"),
            end_time=query.get("endTime"),
        )
        if not items:
            return {"list": [], "total": total}
        benefits = await member_benefit_repository.list_all(db)
        benefit_map = {b.level_code: b for b in benefits}
        list_data = [
            {
                **_build_package_list_vo(pkg),
                "status": pkg.status,
                "createTime": _format_dt(pkg.create_time),
            }
            for pkg in items
        ]
        return {"list": list_data, "total": total}

    async def get_form(self, db: AsyncSession, package_id: int) -> dict:
        pkg = await package_repository.get_by_id(db, package_id)
        if not pkg:
            raise BusinessException(ResultCode.PACKAGE_NOT_FOUND)
        return {
            "id": pkg.id,
            "name": pkg.name,
            "packageType": pkg.package_type,
            "levelCode": pkg.level_code,
            "period": pkg.period,
            "periodDays": pkg.period_days,
            "creditAmount": pkg.credit_amount,
            "originalPrice": pkg.original_price,
            "salePrice": pkg.sale_price,
            "description": pkg.description,
            "benefitOverrides": pkg.benefit_overrides,
            "sort": pkg.sort,
            "status": pkg.status,
        }

    async def create(self, db: AsyncSession, form: dict) -> None:
        existing = await package_repository.get_by_name(db, form["name"])
        if existing:
            raise BusinessException(ResultCode.DATA_EXISTS, "套餐名称已被历史记录占用")
        package_type = form.get("packageType", "vip")
        _validate_package_form(form, package_type)

        if package_type == "credit":
            credit_amount = form.get("creditAmount")
            level_code = None
            period = None
            period_days = None
        else:
            credit_amount = None
            level_code = form.get("levelCode")
            period = form.get("period")
            period_days = form.get("periodDays")

        pkg = SysPackage(
            name=form["name"],
            package_type=package_type,
            level_code=level_code,
            period=period,
            period_days=period_days,
            credit_amount=credit_amount,
            original_price=form["originalPrice"],
            sale_price=form["salePrice"],
            description=form.get("description"),
            benefit_overrides=form.get("benefitOverrides"),
            sort=form.get("sort", 0),
            status=form.get("status", 0),
            sales_count=0,
        )
        await package_repository.create(db, pkg)
        await _invalidate_package_cache()

    async def update(self, db: AsyncSession, package_id: int, form: dict) -> None:
        pkg = await package_repository.get_by_id(db, package_id)
        if not pkg:
            raise BusinessException(ResultCode.PACKAGE_NOT_FOUND)
        # 商品类型创建后锁定，以库中记录为准，请求携带的 packageType 被忽略
        package_type = pkg.package_type
        _validate_package_form(form, package_type)
        if pkg.name != form["name"]:
            dup = await package_repository.get_by_name(db, form["name"])
            if dup and dup.id != package_id:
                raise BusinessException(ResultCode.DATA_EXISTS, "套餐名称已被历史记录占用")
        pkg.name = form["name"]
        if package_type == "credit":
            pkg.credit_amount = form.get("creditAmount")
            pkg.level_code = None
            pkg.period = None
            pkg.period_days = None
        else:
            pkg.level_code = form["levelCode"]
            pkg.period = form["period"]
            pkg.period_days = form["periodDays"]
        pkg.original_price = form["originalPrice"]
        pkg.sale_price = form["salePrice"]
        pkg.description = form.get("description")
        pkg.benefit_overrides = form.get("benefitOverrides")
        if form.get("sort") is not None:
            pkg.sort = form["sort"]
        await db.flush()
        await _invalidate_package_cache(package_id)

    async def update_status(self, db: AsyncSession, package_id: int, status: int) -> None:
        pkg = await package_repository.get_by_id(db, package_id)
        if not pkg:
            raise BusinessException(ResultCode.PACKAGE_NOT_FOUND)
        if status == 0:
            active_promos = await promotion_repository.list_active_by_package_id(db, package_id)
            if active_promos:
                raise BusinessException(ResultCode.PACKAGE_IN_PROMOTION)
        pkg.status = status
        await db.flush()
        await _invalidate_package_cache(package_id)

    async def delete_by_ids(self, db: AsyncSession, ids: list[int]) -> None:
        for package_id in ids:
            pkg = await package_repository.get_by_id(db, package_id)
            if not pkg:
                raise BusinessException(ResultCode.PACKAGE_NOT_FOUND)
            order_count_stmt = (
                select(func.count())
                .select_from(SysOrder)
                .where(
                    SysOrder.package_id == package_id,
                    SysOrder.deleted == 0,
                )
            )
            order_count = (await db.execute(order_count_stmt)).scalar() or 0
            if order_count > 0:
                raise BusinessException(ResultCode.PACKAGE_HAS_ORDERS)
        await package_repository.soft_delete_by_ids(db, ids)
        await _invalidate_package_cache()

    async def calculate_price(
        self,
        db: AsyncSession,
        package_id: int,
        user_coupon_id: int | None,
        user_id: int | None = None,
    ) -> dict:
        pkg = await package_repository.get_by_id(db, package_id)
        if not pkg:
            raise BusinessException(ResultCode.PACKAGE_NOT_FOUND)

        sale_price = pkg.sale_price
        discount_amount = 0

        active_promos = await promotion_repository.list_active_by_package_id(db, package_id)
        for item in active_promos:
            pp = item["promotion_package"]
            if pp.discount_type == "percent":
                discount_amount = max(discount_amount, sale_price * pp.discount_value // 100)
            elif pp.discount_type == "fixed":
                discount_amount = max(discount_amount, pp.discount_value)
            elif pp.discount_type == "full_reduction":
                rules = item["promotion"].activity_rules
                if isinstance(rules, dict):
                    tiers = rules.get("tiers") or []
                    matched = [t for t in tiers if sale_price >= int(t.get("threshold", 0))]
                    if matched:
                        discount_amount = max(
                            discount_amount,
                            max(int(t.get("faceValue", 0)) for t in matched),
                        )

        promo_new_user_only = any(
            item["promotion"].new_user_only == 1 for item in active_promos
        )
        if promo_new_user_only and user_id is not None:
            paid_stmt = select(func.count()).select_from(SysOrder).where(
                SysOrder.user_id == user_id,
                SysOrder.status.in_([2, 3]),
                SysOrder.deleted == 0,
            )
            has_paid = int((await db.execute(paid_stmt)).scalar() or 0) > 0
            if has_paid:
                raise BusinessException(ResultCode.BUSINESS_ERROR, "该套餐仅限新用户购买")

        coupon_amount = 0
        if user_coupon_id:
            user_coupon = await user_coupon_repository.get_by_id(db, user_coupon_id)
            if not user_coupon:
                raise BusinessException(ResultCode.COUPON_NOT_FOUND)
            if user_id and user_coupon.user_id != user_id:
                raise BusinessException(ResultCode.COUPON_NOT_FOUND)
            if user_coupon.status not in (1, 4):
                raise BusinessException(ResultCode.COUPON_STATUS_INVALID)
            if user_coupon.expire_time and user_coupon.expire_time < datetime.now():
                raise BusinessException(ResultCode.COUPON_EXPIRED)

            coupon = await db.get(SysCoupon, user_coupon.coupon_id)
            if not coupon or coupon.status != 1:
                raise BusinessException(ResultCode.COUPON_NOT_FOUND)

            if coupon.applicable_scope:
                applicable = False
                for scope in coupon.applicable_scope:
                    if isinstance(scope, int) and scope == package_id:
                        applicable = True
                        break
                    if isinstance(scope, str) and scope == pkg.package_type:
                        applicable = True
                        break
                if not applicable:
                    raise BusinessException(ResultCode.COUPON_NOT_APPLICABLE)

            # 体验券直接激活会员卡权益、不产生订单，不参与下单价格计算
            if coupon.type == "trial":
                raise BusinessException(ResultCode.BUSINESS_ERROR, "体验券不参与价格计算，请通过激活流程使用")

            base_price = sale_price - discount_amount
            if coupon.type == "full_reduction":
                if base_price >= (coupon.threshold or 0):
                    coupon_amount = coupon.face_value
            elif coupon.type == "discount":
                coupon_amount = base_price * (100 - coupon.face_value) // 100
            elif coupon.type == "no_threshold":
                coupon_amount = coupon.face_value

        payable_amount = max(0, sale_price - discount_amount - coupon_amount)
        return {
            "originalPrice": pkg.original_price,
            "discountAmount": discount_amount,
            "couponAmount": coupon_amount,
            "payableAmount": payable_amount,
        }

    async def get_sales_stats(self, db: AsyncSession) -> dict:
        total_sales_stmt = (
            select(func.count())
            .select_from(SysOrder)
            .where(
                SysOrder.deleted == 0,
                SysOrder.status.in_([2, 3]),
            )
        )
        total_sales = int((await db.execute(total_sales_stmt)).scalar() or 0)

        revenue_stmt = select(func.coalesce(func.sum(SysOrder.paid_amount), 0)).where(
            SysOrder.deleted == 0,
            SysOrder.status.in_([2, 3]),
        )
        total_revenue = int((await db.execute(revenue_stmt)).scalar() or 0)

        package_stats_stmt = (
            select(
                SysOrder.package_id,
                SysOrder.package_name,
                func.count().label("count"),
                func.coalesce(func.sum(SysOrder.paid_amount), 0).label("revenue"),
            )
            .where(SysOrder.deleted == 0, SysOrder.status.in_([2, 3]))
            .group_by(SysOrder.package_id, SysOrder.package_name)
        )
        package_rows = (await db.execute(package_stats_stmt)).all()
        package_stats = [
            {
                "packageId": row.package_id,
                "packageName": row.package_name,
                "salesCount": row.count,
                "revenue": int(row.revenue),
            }
            for row in package_rows
        ]

        level_stats_stmt = (
            select(
                SysOrder.package_level,
                func.count().label("count"),
                func.coalesce(func.sum(SysOrder.paid_amount), 0).label("revenue"),
            )
            .where(
                SysOrder.deleted == 0,
                SysOrder.status.in_([2, 3]),
                SysOrder.package_level.isnot(None),
                SysOrder.package_level != "",
            )
            .group_by(SysOrder.package_level)
        )
        level_rows = (await db.execute(level_stats_stmt)).all()
        benefits = await member_benefit_repository.list_all(db)
        benefit_map = {b.level_code: b for b in benefits}
        level_stats = [
            {
                "levelCode": row.package_level,
                "levelName": benefit_map[row.package_level].level_name
                if row.package_level in benefit_map
                else "",
                "salesCount": row.count,
                "revenue": int(row.revenue),
            }
            for row in level_rows
        ]

        type_stats_stmt = (
            select(
                SysOrder.package_type,
                func.count().label("count"),
                func.coalesce(func.sum(SysOrder.paid_amount), 0).label("revenue"),
            )
            .where(SysOrder.deleted == 0, SysOrder.status.in_([2, 3]))
            .group_by(SysOrder.package_type)
        )
        type_rows = (await db.execute(type_stats_stmt)).all()
        type_stats = [
            {
                "packageType": row.package_type,
                "packageTypeName": PACKAGE_TYPE_NAMES.get(row.package_type, row.package_type),
                "salesCount": int(row.count),
                "revenue": int(row.revenue),
            }
            for row in type_rows
        ]

        period_stats_stmt = (
            select(
                SysPackage.period,
                func.count().label("count"),
                func.coalesce(func.sum(SysOrder.paid_amount), 0).label("revenue"),
            )
            .select_from(SysOrder)
            .join(SysPackage, SysOrder.package_id == SysPackage.id)
            .where(
                SysOrder.deleted == 0,
                SysOrder.status.in_([2, 3]),
                SysPackage.deleted == 0,
                SysPackage.period.isnot(None),
            )
            .group_by(SysPackage.period)
        )
        period_rows = (await db.execute(period_stats_stmt)).all()
        period_stats = [
            {
                "period": row.period,
                "periodName": PERIOD_NAMES.get(row.period, row.period),
                "salesCount": int(row.count),
                "revenue": int(row.revenue),
            }
            for row in period_rows
        ]

        coupon_issued_stmt = select(func.coalesce(func.sum(SysCoupon.issued_qty), 0)).where(
            SysCoupon.deleted == 0
        )
        total_issued = int((await db.execute(coupon_issued_stmt)).scalar() or 0)

        coupon_used_stmt = select(func.coalesce(func.sum(SysCoupon.used_qty), 0)).where(
            SysCoupon.deleted == 0
        )
        total_used = int((await db.execute(coupon_used_stmt)).scalar() or 0)

        usage_rate = (total_used / total_issued) if total_issued > 0 else 0

        return {
            "totalSales": total_sales,
            "totalRevenue": total_revenue,
            "packageStats": package_stats,
            "levelStats": level_stats,
            "typeStats": type_stats,
            "periodStats": period_stats,
            "couponStats": {
                "totalIssued": total_issued,
                "totalUsed": total_used,
                "usageRate": usage_rate,
            },
        }


package_service = PackageService()
