from datetime import datetime
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_coupon import SysCoupon
from app.models.entity.sys_order import SysOrder
from app.models.entity.sys_package import SysPackage
from app.models.entity.sys_user_coupon import SysUserCoupon
from app.models.entity.sys_member_benefit import SysMemberBenefit
from app.models.entity.sys_promotion import SysPromotion
from app.repository.coupon_repository import user_coupon_repository
from app.repository.member_benefit_repository import member_benefit_repository
from app.repository.package_repository import package_repository
from app.repository.promotion_repository import promotion_repository

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
    "monthly": "月包",
    "quarterly": "季包",
    "yearly": "年包",
}


def _format_dt(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _get_effective_benefits(benefit: Optional[SysMemberBenefit], overrides: Optional[dict]) -> dict:
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


def _calc_daily_price(sale_price: int, period_days: int) -> int:
    if period_days <= 0:
        return 0
    return sale_price // period_days


def _build_package_detail(db_pkg: SysPackage, benefit: Optional[SysMemberBenefit]) -> dict:
    overrides = db_pkg.benefit_overrides if isinstance(db_pkg.benefit_overrides, dict) else None
    benefits = _get_effective_benefits(benefit, overrides)
    return {
        "id": db_pkg.id,
        "name": db_pkg.name,
        "levelCode": db_pkg.level_code,
        "levelName": benefit.level_name if benefit else "",
        "period": db_pkg.period,
        "periodDays": db_pkg.period_days,
        "originalPrice": db_pkg.original_price,
        "salePrice": db_pkg.sale_price,
        "dailyPrice": _calc_daily_price(db_pkg.sale_price, db_pkg.period_days),
        "description": db_pkg.description,
        "benefits": benefits,
        "salesCount": db_pkg.sales_count,
    }


class PackageService:

    @staticmethod
    async def list_on_sale(db: AsyncSession) -> list[dict]:
        packages = await package_repository.list_on_sale(db)
        if not packages:
            return []
        benefits = await member_benefit_repository.list_all(db)
        benefit_map = {b.level_code: b for b in benefits}
        result = []
        for pkg in packages:
            benefit = benefit_map.get(pkg.level_code)
            result.append(_build_package_detail(pkg, benefit))
        return result

    @staticmethod
    async def get_detail(db: AsyncSession, package_id: int) -> dict:
        pkg = await package_repository.get_by_id(db, package_id)
        if not pkg:
            raise BusinessException(ResultCode.PACKAGE_NOT_FOUND)
        benefit = await member_benefit_repository.get_by_level_code(db, pkg.level_code)
        detail = _build_package_detail(pkg, benefit)

        active_promos = await promotion_repository.list_active_by_package_id(db, package_id)
        if active_promos:
            detail["activePromotions"] = [
                _promotion_to_vo(item["promotion"]) for item in active_promos
            ]
        return detail

    @staticmethod
    async def get_page(db: AsyncSession, query: dict) -> dict:
        items, total = await package_repository.get_page(
            db,
            query["pageNum"],
            query["pageSize"],
            name=query.get("name"),
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
                "id": pkg.id,
                "name": pkg.name,
                "levelCode": pkg.level_code,
                "levelName": benefit_map[pkg.level_code].level_name if pkg.level_code in benefit_map else "",
                "period": pkg.period,
                "periodDays": pkg.period_days,
                "originalPrice": pkg.original_price,
                "salePrice": pkg.sale_price,
                "dailyPrice": _calc_daily_price(pkg.sale_price, pkg.period_days),
                "salesCount": pkg.sales_count,
                "status": pkg.status,
                "createTime": _format_dt(pkg.create_time),
            }
            for pkg in items
        ]
        return {"list": list_data, "total": total}

    @staticmethod
    async def get_form(db: AsyncSession, package_id: int) -> dict:
        pkg = await package_repository.get_by_id(db, package_id)
        if not pkg:
            raise BusinessException(ResultCode.PACKAGE_NOT_FOUND)
        return {
            "id": pkg.id,
            "name": pkg.name,
            "levelCode": pkg.level_code,
            "period": pkg.period,
            "periodDays": pkg.period_days,
            "originalPrice": pkg.original_price,
            "salePrice": pkg.sale_price,
            "description": pkg.description,
            "benefitOverrides": pkg.benefit_overrides,
            "sort": pkg.sort,
            "status": pkg.status,
        }

    @staticmethod
    async def create(db: AsyncSession, form: dict) -> None:
        existing = await package_repository.get_by_name(db, form["name"])
        if existing:
            raise BusinessException(ResultCode.DATA_EXISTS, "套餐名称已存在")
        pkg = SysPackage(
            name=form["name"],
            level_code=form["levelCode"],
            period=form["period"],
            period_days=form["periodDays"],
            original_price=form["originalPrice"],
            sale_price=form["salePrice"],
            description=form.get("description"),
            benefit_overrides=form.get("benefitOverrides"),
            sort=form.get("sort", 0),
            status=form.get("status", 0),
        )
        await package_repository.create(db, pkg)

    @staticmethod
    async def update(db: AsyncSession, package_id: int, form: dict) -> None:
        pkg = await package_repository.get_by_id(db, package_id)
        if not pkg:
            raise BusinessException(ResultCode.PACKAGE_NOT_FOUND)
        pkg.name = form["name"]
        pkg.level_code = form["levelCode"]
        pkg.period = form["period"]
        pkg.period_days = form["periodDays"]
        pkg.original_price = form["originalPrice"]
        pkg.sale_price = form["salePrice"]
        pkg.description = form.get("description")
        pkg.benefit_overrides = form.get("benefitOverrides")
        pkg.sort = form.get("sort", 0)
        if form.get("status") is not None:
            pkg.status = form["status"]
        await db.flush()

    @staticmethod
    async def update_status(db: AsyncSession, package_id: int, status: int) -> None:
        pkg = await package_repository.get_by_id(db, package_id)
        if not pkg:
            raise BusinessException(ResultCode.PACKAGE_NOT_FOUND)
        pkg.status = status
        await db.flush()

    @staticmethod
    async def delete_by_ids(db: AsyncSession, ids: list[int]) -> None:
        for package_id in ids:
            pkg = await package_repository.get_by_id(db, package_id)
            if not pkg:
                raise BusinessException(ResultCode.PACKAGE_NOT_FOUND)
        await package_repository.soft_delete_by_ids(db, ids)

    @staticmethod
    async def calculate_price(
        db: AsyncSession,
        package_id: int,
        user_coupon_id: Optional[int],
        user_id: Optional[int] = None,
    ) -> dict:
        pkg = await package_repository.get_by_id(db, package_id)
        if not pkg:
            raise BusinessException(ResultCode.PACKAGE_NOT_FOUND)

        original_price = pkg.sale_price
        discount_amount = 0

        active_promos = await promotion_repository.list_active_by_package_id(db, package_id)
        for item in active_promos:
            pp = item["promotion_package"]
            if pp.discount_type == "percent":
                discount_amount = max(discount_amount, original_price * pp.discount_value // 100)
            else:
                discount_amount = max(discount_amount, pp.discount_value)

        coupon_amount = 0
        if user_coupon_id:
            user_coupon = await user_coupon_repository.get_by_id(db, user_coupon_id)
            if not user_coupon:
                raise BusinessException(ResultCode.COUPON_NOT_FOUND)
            if user_id and user_coupon.user_id != user_id:
                raise BusinessException(ResultCode.BUSINESS_ERROR, "优惠券不属于当前用户")
            if user_coupon.status != 1:
                raise BusinessException(ResultCode.COUPON_STATUS_INVALID)
            if user_coupon.expire_time and user_coupon.expire_time < datetime.now():
                raise BusinessException(ResultCode.COUPON_EXPIRED)

            coupon = await db.get(SysCoupon, user_coupon.coupon_id)
            if not coupon:
                raise BusinessException(ResultCode.COUPON_NOT_FOUND)

            if coupon.applicable_scope:
                if package_id not in coupon.applicable_scope:
                    raise BusinessException(ResultCode.COUPON_NOT_APPLICABLE)

            base_price = original_price - discount_amount
            if coupon.type == "full_reduction":
                if base_price >= (coupon.threshold or 0):
                    coupon_amount = coupon.face_value
            elif coupon.type == "discount":
                coupon_amount = base_price * (100 - coupon.face_value) // 100
            elif coupon.type == "no_threshold":
                coupon_amount = coupon.face_value
            elif coupon.type == "trial":
                coupon_amount = base_price

        payable_amount = max(0, original_price - discount_amount - coupon_amount)
        return {
            "originalPrice": original_price,
            "discountAmount": discount_amount,
            "couponAmount": coupon_amount,
            "payableAmount": payable_amount,
        }

    @staticmethod
    async def get_sales_stats(db: AsyncSession) -> dict:
        total_sales_stmt = select(func.coalesce(func.sum(SysPackage.sales_count), 0)).where(
            SysPackage.deleted == 0
        )
        total_sales = (await db.execute(total_sales_stmt)).scalar() or 0

        revenue_stmt = select(
            func.coalesce(func.sum(SysOrder.paid_amount), 0)
        ).where(
            SysOrder.deleted == 0,
            SysOrder.status.in_([2, 3]),
        )
        total_revenue = (await db.execute(revenue_stmt)).scalar() or 0

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
                "revenue": row.revenue,
            }
            for row in package_rows
        ]

        level_stats_stmt = (
            select(
                SysOrder.package_level,
                func.count().label("count"),
                func.coalesce(func.sum(SysOrder.paid_amount), 0).label("revenue"),
            )
            .where(SysOrder.deleted == 0, SysOrder.status.in_([2, 3]))
            .group_by(SysOrder.package_level)
        )
        level_rows = (await db.execute(level_stats_stmt)).all()
        benefits = await member_benefit_repository.list_all(db)
        benefit_map = {b.level_code: b for b in benefits}
        level_stats = [
            {
                "levelCode": row.package_level,
                "levelName": benefit_map[row.package_level].level_name if row.package_level in benefit_map else "",
                "salesCount": row.count,
                "revenue": row.revenue,
            }
            for row in level_rows
        ]

        period_stats_stmt = (
            select(
                SysPackage.period,
                func.coalesce(func.sum(SysPackage.sales_count), 0).label("count"),
                func.coalesce(func.sum(SysPackage.sales_count * SysPackage.sale_price), 0).label("revenue"),
            )
            .where(SysPackage.deleted == 0)
            .group_by(SysPackage.period)
        )
        period_rows = (await db.execute(period_stats_stmt)).all()
        period_stats = [
            {
                "period": row.period,
                "periodName": PERIOD_NAMES.get(row.period, row.period),
                "salesCount": row.count,
                "revenue": row.revenue,
            }
            for row in period_rows
        ]

        coupon_issued_stmt = select(func.coalesce(func.sum(SysCoupon.issued_qty), 0)).where(
            SysCoupon.deleted == 0
        )
        total_issued = (await db.execute(coupon_issued_stmt)).scalar() or 0

        coupon_used_stmt = select(func.coalesce(func.sum(SysCoupon.used_qty), 0)).where(
            SysCoupon.deleted == 0
        )
        total_used = (await db.execute(coupon_used_stmt)).scalar() or 0

        usage_rate = (total_used / total_issued) if total_issued > 0 else 0

        return {
            "totalSales": total_sales,
            "totalRevenue": total_revenue,
            "packageStats": package_stats,
            "levelStats": level_stats,
            "periodStats": period_stats,
            "couponStats": {
                "totalIssued": total_issued,
                "totalUsed": total_used,
                "usageRate": usage_rate,
            },
        }
