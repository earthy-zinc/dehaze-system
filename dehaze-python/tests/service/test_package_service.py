"""
套餐管理 service 单元测试（真实 MySQL 测试库 dehaze_test + SAVEPOINT 回滚）。

覆盖 package_service 的双类型（vip 会员卡 / credit 积分卡）核心逻辑：
创建、更新、列表/详情、上下架、删除、calculate_price 价格计算、get_sales_stats 销量统计。

遵循 dehaze 测试规范：
- 仅依赖 db fixture（外部事务 + SAVEPOINT 回滚）与 mock_redis（autouse）
- 只断言业务结果，不 mock 调用序列
- 命名 test_功能_场景
"""

import pytest
from datetime import datetime, timedelta

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_order import SysOrder
from app.models.entity.sys_package import SysPackage
from app.models.schema.promotion import PromotionForm, PromotionPackageForm
from app.repository.package_repository import package_repository
from app.repository.promotion_repository import promotion_repository
from app.service.package_service import package_service
from app.service.promotion_service import promotion_service

pytestmark = pytest.mark.requires_db

USER_ID = 1000001


def _daily_price(sale_price: int, period_days: int) -> int:
    if period_days <= 0:
        return 0
    return (2 * sale_price + period_days) // (2 * period_days)


async def _create_pkg(db, data: dict) -> SysPackage:
    await package_service.create(db, data)
    return await package_repository.get_by_name(db, data["name"])


def _promo_payload(name: str, start, end, extra: dict | None = None) -> dict:
    base = {
        "name": name,
        "type": "full_reduction",
        "startTime": start,
        "endTime": end,
        "activityRules": extra.pop("activityRules", {"tiers": [{"threshold": 5000, "faceValue": 500}]}) if extra else {"tiers": [{"threshold": 5000, "faceValue": 500}]},
        "newUserOnly": 0,
        "status": 1,
    }
    if extra:
        base.update(extra)
    return base


async def _bind_promo(db, pkg_id: int, discount_type: str, discount_value: int, tiers: list | None = None):
    from datetime import datetime, timedelta
    activity_rules = {"discount_type": discount_type, "discount_value": discount_value}
    if tiers is not None:
        activity_rules["tiers"] = tiers
    now = datetime.now()
    promo = await promotion_service.create(db, PromotionForm(**_promo_payload(
        f"test_promo_{pkg_id}_{discount_type}",
        (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
        (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
        extra={"activityRules": activity_rules},
    )))
    await promotion_service.bind_packages(
        db, promo["id"], PromotionPackageForm(packageIds=[pkg_id])
    )
    return promo


# ===================== 创建 =====================

async def test_create_vip_success(db):
    pkg = await _create_pkg(db, {
        "name": "test_pkg_vip_ok",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    assert pkg.id is not None
    assert pkg.package_type == "vip"
    assert pkg.sales_count == 0
    assert pkg.status == 0
    assert pkg.level_code == "level_1"
    assert pkg.period == "monthly"
    assert pkg.period_days == 30
    assert pkg.credit_amount is None


async def test_create_credit_success(db):
    pkg = await _create_pkg(db, {
        "name": "test_pkg_credit_ok",
        "packageType": "credit",
        "creditAmount": 1000,
        "originalPrice": 5000,
        "salePrice": 4000,
    })
    assert pkg.package_type == "credit"
    assert pkg.credit_amount == 1000
    assert pkg.level_code is None
    assert pkg.period is None
    assert pkg.period_days is None
    assert pkg.sales_count == 0
    assert pkg.status == 0


async def test_create_invalid_package_type(db):
    with pytest.raises(BusinessException) as exc:
        await package_service.create(db, {
            "name": "test_pkg_badtype",
            "packageType": "unknown",
            "originalPrice": 5000,
            "salePrice": 4000,
        })
    assert exc.value.code == ResultCode.PARAM_ERROR


async def test_create_sale_gt_original(db):
    with pytest.raises(BusinessException) as exc:
        await package_service.create(db, {
            "name": "test_pkg_price",
            "packageType": "vip",
            "levelCode": "level_1",
            "period": "monthly",
            "periodDays": 30,
            "originalPrice": 4000,
            "salePrice": 5000,
        })
    assert exc.value.code == ResultCode.PARAM_ERROR


async def test_create_credit_zero_amount(db):
    with pytest.raises(BusinessException) as exc:
        await package_service.create(db, {
            "name": "test_pkg_credit_zero",
            "packageType": "credit",
            "creditAmount": 0,
            "originalPrice": 5000,
            "salePrice": 4000,
        })
    assert exc.value.code == ResultCode.PARAM_ERROR


async def test_create_vip_missing_level(db):
    with pytest.raises(BusinessException) as exc:
        await package_service.create(db, {
            "name": "test_pkg_vip_mlv",
            "packageType": "vip",
            "period": "monthly",
            "periodDays": 30,
            "originalPrice": 5000,
            "salePrice": 4000,
        })
    assert exc.value.code == ResultCode.PARAM_ERROR


async def test_create_vip_invalid_period(db):
    with pytest.raises(BusinessException) as exc:
        await package_service.create(db, {
            "name": "test_pkg_vip_pd",
            "packageType": "vip",
            "levelCode": "level_1",
            "period": "decade",
            "periodDays": 30,
            "originalPrice": 5000,
            "salePrice": 4000,
        })
    assert exc.value.code == ResultCode.PARAM_ERROR


async def test_create_duplicate_with_deleted(db):
    dup = await _create_pkg(db, {
        "name": "test_pkg_dup_soft",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    await package_service.delete_by_ids(db, [dup.id])
    with pytest.raises(BusinessException) as exc:
        await package_service.create(db, {
            "name": "test_pkg_dup_soft",
            "packageType": "vip",
            "levelCode": "level_1",
            "period": "monthly",
            "periodDays": 30,
            "originalPrice": 10000,
            "salePrice": 8000,
        })
    assert exc.value.code == ResultCode.DATA_EXISTS


# ===================== 更新 =====================

async def test_update_type_locked(db):
    pkg = await _create_pkg(db, {
        "name": "test_pkg_vip_tl",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    await package_service.update(db, pkg.id, {
        "packageType": "credit",
        "creditAmount": 500,
        "salePrice": 7000,
        "name": "test_pkg_vip_tl",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
    })
    upd = await package_repository.get_by_id(db, pkg.id)
    assert upd.package_type == "vip"
    assert upd.credit_amount is None
    assert upd.sale_price == 7000


async def test_update_credit_amount(db):
    pkg = await _create_pkg(db, {
        "name": "test_pkg_credit_ua",
        "packageType": "credit",
        "creditAmount": 1000,
        "originalPrice": 5000,
        "salePrice": 4000,
    })
    await package_service.update(db, pkg.id, {
        "name": "test_pkg_credit_ua",
        "packageType": "credit",
        "creditAmount": 2000,
        "originalPrice": 5000,
        "salePrice": 4000,
    })
    upd = await package_repository.get_by_id(db, pkg.id)
    assert upd.credit_amount == 2000


async def test_update_rename_occupied(db):
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_ro",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    credit = await _create_pkg(db, {
        "name": "test_pkg_credit_ro",
        "packageType": "credit",
        "creditAmount": 1000,
        "originalPrice": 5000,
        "salePrice": 4000,
    })
    with pytest.raises(BusinessException) as exc:
        await package_service.update(db, vip.id, {
            "name": credit.name,
            "packageType": "vip",
            "levelCode": "level_1",
            "period": "monthly",
            "periodDays": 30,
            "originalPrice": 10000,
            "salePrice": 8000,
        })
    assert exc.value.code == ResultCode.DATA_EXISTS


async def test_update_carries_status_unchanged(db):
    pkg = await _create_pkg(db, {
        "name": "test_pkg_vip_su",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    await package_service.update(db, pkg.id, {
        "name": "test_pkg_vip_su",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 6000,
        "status": 1,
    })
    upd = await package_repository.get_by_id(db, pkg.id)
    assert upd.status == 0


# ===================== 列表 / 详情 =====================

async def test_list_on_sale_filter_by_type(db, mock_redis):
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_lf",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    credit = await _create_pkg(db, {
        "name": "test_pkg_credit_lf",
        "packageType": "credit",
        "creditAmount": 1000,
        "originalPrice": 5000,
        "salePrice": 4000,
    })
    await package_service.update_status(db, vip.id, 1)
    await package_service.update_status(db, credit.id, 1)
    vip_list = await package_service.list_on_sale(db, "vip")
    assert all(p["packageType"] == "vip" for p in vip_list)
    credit_list = await package_service.list_on_sale(db, "credit")
    assert all(p["packageType"] == "credit" for p in credit_list)


async def test_list_on_sale_vip_daily_price(db):
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_dp",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    await package_service.update_status(db, vip.id, 1)
    items = await package_service.list_on_sale(db, "vip")
    v = next(p for p in items if p["id"] == vip.id)
    assert v["dailyPrice"] == _daily_price(8000, 30)


async def test_list_on_sale_credit_unit_price(db):
    credit = await _create_pkg(db, {
        "name": "test_pkg_credit_up",
        "packageType": "credit",
        "creditAmount": 1000,
        "originalPrice": 5000,
        "salePrice": 4000,
    })
    await package_service.update_status(db, credit.id, 1)
    items = await package_service.list_on_sale(db, "credit")
    c = next(p for p in items if p["id"] == credit.id)
    assert c["creditAmount"] == 1000
    assert c["creditUnitPrice"] == 4000 // 1000


async def test_off_shelf_not_in_onsale(db):
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_os",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    items = await package_service.list_on_sale(db, None)
    assert vip.id not in [p["id"] for p in items]


async def test_get_detail_off_shelf(db):
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_gd",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    with pytest.raises(BusinessException) as exc:
        await package_service.get_detail(db, vip.id)
    assert exc.value.code == ResultCode.PACKAGE_OFF_SHELF


async def test_get_detail_not_found(db):
    with pytest.raises(BusinessException) as exc:
        await package_service.get_detail(db, 999999999)
    assert exc.value.code == ResultCode.PACKAGE_NOT_FOUND


async def test_get_detail_has_active_promotions(db):
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_ap",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    await package_service.update_status(db, vip.id, 1)
    detail = await package_service.get_detail(db, vip.id)
    assert detail["activePromotions"] == []


# ===================== 上下架 =====================

async def test_off_shelf_blocked_by_promotion(db):
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_bp",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    await _bind_promo(db, vip.id, "percent", 10)
    with pytest.raises(BusinessException) as exc:
        await package_service.update_status(db, vip.id, 0)
    assert exc.value.code == ResultCode.PACKAGE_IN_PROMOTION


async def test_off_shelf_after_promotion_ended(db):
    from datetime import datetime, timedelta
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_pe",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    now = datetime.now()
    promo = await promotion_service.create(db, PromotionForm(**_promo_payload(
        "test_promo_ended",
        (now - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S"),
        (now - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S"),
    )))
    await promotion_service.bind_packages(
        db, promo["id"], PromotionPackageForm(packageIds=[vip.id])
    )
    await package_service.update_status(db, vip.id, 0)
    assert (await package_repository.get_by_id(db, vip.id)).status == 0


# ===================== 删除 =====================

async def test_delete_with_orders(db):
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_do",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    db.add(SysOrder(
        order_no="test_order_x1", user_id=USER_ID, package_id=vip.id,
        package_level="level_1", package_name=vip.name, package_type="vip",
        original_price=8000, payable_amount=8000, paid_amount=8000,
        pay_method="test", status=2, expire_time=datetime.now(),
    ))
    await db.flush()
    with pytest.raises(BusinessException) as exc:
        await package_service.delete_by_ids(db, [vip.id])
    assert exc.value.code == ResultCode.PACKAGE_HAS_ORDERS


async def test_delete_no_orders(db):
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_dn",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    await package_service.delete_by_ids(db, [vip.id])
    assert (await package_repository.get_by_id(db, vip.id)).deleted == 1


# ===================== calculate_price =====================

async def test_calc_no_promo_no_coupon(db):
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_c0",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    await package_service.update_status(db, vip.id, 1)
    result = await package_service.calculate_price(db, vip.id, None, user_id=USER_ID)
    assert result["payableAmount"] == 8000


async def test_calc_percent_discount(db):
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_c1",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    await _bind_promo(db, vip.id, "percent", 20)
    await package_service.update_status(db, vip.id, 1)
    result = await package_service.calculate_price(db, vip.id, None, user_id=USER_ID)
    # 8000 * (100-20)% = 6400
    assert result["payableAmount"] == 6400


async def test_calc_fixed_discount(db):
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_c2",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    await _bind_promo(db, vip.id, "fixed", 3000)
    await package_service.update_status(db, vip.id, 1)
    result = await package_service.calculate_price(db, vip.id, None, user_id=USER_ID)
    # (8000-3000)=5000
    assert result["payableAmount"] == 5000


async def test_calc_max_discount_not_stack(db):
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_c3",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    await _bind_promo(db, vip.id, "percent", 20)  # 折后 6400
    await _bind_promo(db, vip.id, "fixed", 1000)  # 减后 7000；取最低 6400
    await package_service.update_status(db, vip.id, 1)
    result = await package_service.calculate_price(db, vip.id, None, user_id=USER_ID)
    assert result["payableAmount"] == 6400


async def test_calc_full_reduction_tiers(db):
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_c4",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    await _bind_promo(db, vip.id, "full_reduction", 0,
                      tiers=[{"threshold": 5000, "faceValue": 500}, {"threshold": 8000, "faceValue": 1500}])
    await package_service.update_status(db, vip.id, 1)
    # sale_price=8000 命中 >=8000 档，减 1500 → 6500
    result = await package_service.calculate_price(db, vip.id, None, user_id=USER_ID)
    assert result["payableAmount"] == 6500


async def test_calc_full_reduction_below_threshold(db):
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_c5",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    await _bind_promo(db, vip.id, "full_reduction", 0,
                      tiers=[{"threshold": 10000, "faceValue": 500}])
    await package_service.update_status(db, vip.id, 1)
    # sale_price=8000 < 10000 未达门槛 → 不生效 → 8000
    result = await package_service.calculate_price(db, vip.id, None, user_id=USER_ID)
    assert result["payableAmount"] == 8000


async def test_calc_new_user_only_blocked(db):
    from datetime import datetime, timedelta
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_c6",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    db.add(SysOrder(
        order_no="test_order_hist1", user_id=USER_ID, package_id=vip.id,
        package_level="level_1", package_name=vip.name, package_type="vip",
        original_price=8000, payable_amount=8000, paid_amount=8000,
        pay_method="test", status=2, expire_time=datetime.now(),
    ))
    await db.flush()
    now = datetime.now()
    promo = await promotion_service.create(db, PromotionForm(**_promo_payload(
        "test_promo_nu",
        (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
        (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
        extra={"newUserOnly": 1},
    )))
    await promotion_service.bind_packages(
        db, promo["id"], PromotionPackageForm(packageIds=[vip.id])
    )
    await package_service.update_status(db, vip.id, 1)
    with pytest.raises(BusinessException) as exc:
        await package_service.calculate_price(db, vip.id, None, user_id=USER_ID)
    assert exc.value.code == ResultCode.BUSINESS_ERROR
    assert "新用户" in exc.value.message


# ===================== calculate_price + 优惠券 =====================

async def test_calc_coupon_discount(db):
    from app.models.entity.sys_coupon import SysCoupon
    from app.repository.coupon_repository import coupon_repository
    from app.service.coupon_service import coupon_service
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_cc",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    coupon = SysCoupon(
        name="测试券-折扣", type="discount", face_value=20, threshold=0,
        valid_type="relative", valid_days=30, total_qty=-1, per_user_limit=5,
        applicable_scope=[], status=1,
    )
    await coupon_repository.create(db, coupon)
    received = await coupon_service.receive(db, coupon.id, USER_ID)
    await _bind_promo(db, vip.id, "percent", 10)  # 折后 7200
    await package_service.update_status(db, vip.id, 1)
    # promo 折后 7200，券再 8 折抵扣 5760 → 应付 8000-800-5760=1440
    result = await package_service.calculate_price(
        db, vip.id, received["userCouponId"], user_id=USER_ID)
    assert result["payableAmount"] == 1440


async def test_calc_coupon_not_applicable(db):
    from app.models.entity.sys_coupon import SysCoupon
    from app.repository.coupon_repository import coupon_repository
    from app.service.coupon_service import coupon_service
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_cna",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    credit = await _create_pkg(db, {
        "name": "test_pkg_credit_cna",
        "packageType": "credit",
        "creditAmount": 1000,
        "originalPrice": 5000,
        "salePrice": 4000,
    })
    coupon = SysCoupon(
        name="测试券-限定", type="no_threshold", face_value=1000, threshold=0,
        valid_type="relative", valid_days=30, total_qty=-1, per_user_limit=5,
        applicable_scope=[credit.id], status=1,
    )
    await coupon_repository.create(db, coupon)
    received = await coupon_service.receive(db, coupon.id, USER_ID)
    await package_service.update_status(db, vip.id, 1)
    with pytest.raises(BusinessException) as exc:
        await package_service.calculate_price(
            db, vip.id, received["userCouponId"], user_id=USER_ID)
    assert exc.value.code == ResultCode.COUPON_NOT_APPLICABLE


async def test_calc_coupon_status_invalid(db):
    from app.models.entity.sys_coupon import SysCoupon
    from app.repository.coupon_repository import coupon_repository, user_coupon_repository
    from app.service.coupon_service import coupon_service
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_csi",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    coupon = SysCoupon(
        name="测试券-失效", type="no_threshold", face_value=1000, threshold=0,
        valid_type="relative", valid_days=30, total_qty=-1, per_user_limit=5,
        applicable_scope=[], status=1,
    )
    await coupon_repository.create(db, coupon)
    received = await coupon_service.receive(db, coupon.id, USER_ID)
    # 模拟已领取的券状态变为无效（如已使用/已过期）
    uc = await user_coupon_repository.get_by_id(db, received["userCouponId"])
    uc.status = 0
    await db.flush()
    await package_service.update_status(db, vip.id, 1)
    with pytest.raises(BusinessException) as exc:
        await package_service.calculate_price(
            db, vip.id, received["userCouponId"], user_id=USER_ID)
    assert exc.value.code == ResultCode.COUPON_STATUS_INVALID


async def test_calc_coupon_template_disabled(db):
    from app.models.entity.sys_coupon import SysCoupon
    from app.models.entity.sys_user_coupon import SysUserCoupon
    from app.repository.coupon_repository import coupon_repository, user_coupon_repository
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_cd",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    coupon = SysCoupon(
        name="测试券-禁用", type="no_threshold", face_value=1000, threshold=0,
        valid_type="relative", valid_days=30, total_qty=-1, per_user_limit=5,
        applicable_scope=[], status=0,
    )
    await coupon_repository.create(db, coupon)
    user_coupon = SysUserCoupon(
        user_id=USER_ID, coupon_id=coupon.id, status=1,
        receive_time=datetime.now(), expire_time=datetime.now() + timedelta(days=30),
    )
    await user_coupon_repository.create(db, user_coupon)
    await package_service.update_status(db, vip.id, 1)
    with pytest.raises(BusinessException) as exc:
        await package_service.calculate_price(
            db, vip.id, user_coupon.id, user_id=USER_ID)
    assert exc.value.code == ResultCode.COUPON_NOT_FOUND


async def test_calc_coupon_trial_rejected(db):
    from app.models.entity.sys_coupon import SysCoupon
    from app.repository.coupon_repository import coupon_repository
    from app.service.coupon_service import coupon_service
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_ctf",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    coupon = SysCoupon(
        name="测试券-体验", type="trial", face_value=0, threshold=0,
        valid_type="relative", valid_days=3, total_qty=-1, per_user_limit=1,
        applicable_scope=[], status=1,
    )
    await coupon_repository.create(db, coupon)
    received = await coupon_service.receive(db, coupon.id, USER_ID)
    await package_service.update_status(db, vip.id, 1)
    # 体验券直接激活权益、不产生订单，不参与下单价格计算
    with pytest.raises(BusinessException) as exc:
        await package_service.calculate_price(
            db, vip.id, received["userCouponId"], user_id=USER_ID)
    assert exc.value.code == ResultCode.BUSINESS_ERROR
    assert "体验券" in exc.value.message


async def test_calc_coupon_scope_by_package_type(db):
    from app.models.entity.sys_coupon import SysCoupon
    from app.repository.coupon_repository import coupon_repository
    from app.service.coupon_service import coupon_service
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_cs",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    credit = await _create_pkg(db, {
        "name": "test_pkg_credit_cs",
        "packageType": "credit",
        "creditAmount": 1000,
        "originalPrice": 5000,
        "salePrice": 4000,
    })
    vip_coupon = SysCoupon(
        name="测试券-限定会员卡", type="no_threshold", face_value=1000, threshold=0,
        valid_type="relative", valid_days=30, total_qty=-1, per_user_limit=5,
        applicable_scope=["vip"], status=1,
    )
    credit_coupon = SysCoupon(
        name="测试券-限定积分卡", type="no_threshold", face_value=1000, threshold=0,
        valid_type="relative", valid_days=30, total_qty=-1, per_user_limit=5,
        applicable_scope=["credit"], status=1,
    )
    await coupon_repository.create(db, vip_coupon)
    await coupon_repository.create(db, credit_coupon)
    vip_received = await coupon_service.receive(db, vip_coupon.id, USER_ID)
    credit_received = await coupon_service.receive(db, credit_coupon.id, USER_ID)
    await package_service.update_status(db, vip.id, 1)
    await package_service.update_status(db, credit.id, 1)

    vip_result = await package_service.calculate_price(
        db, vip.id, vip_received["userCouponId"], user_id=USER_ID)
    assert vip_result["couponAmount"] == 1000
    assert vip_result["payableAmount"] == 7000

    credit_result = await package_service.calculate_price(
        db, credit.id, credit_received["userCouponId"], user_id=USER_ID)
    assert credit_result["payableAmount"] == 3000

    with pytest.raises(BusinessException) as exc:
        await package_service.calculate_price(
            db, vip.id, credit_received["userCouponId"], user_id=USER_ID)
    assert exc.value.code == ResultCode.COUPON_NOT_APPLICABLE


# ===================== get_sales_stats =====================

async def test_get_sales_stats_type_group(db):
    vip = await _create_pkg(db, {
        "name": "test_pkg_vip_ss",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    })
    credit = await _create_pkg(db, {
        "name": "test_pkg_credit_ss",
        "packageType": "credit",
        "creditAmount": 1000,
        "originalPrice": 5000,
        "salePrice": 4000,
    })
    await package_service.update_status(db, vip.id, 1)
    await package_service.update_status(db, credit.id, 1)
    db.add(SysOrder(
        order_no="test_stat_v1", user_id=USER_ID, package_id=vip.id,
        package_level="level_1", package_name=vip.name, package_type="vip",
        original_price=8000, payable_amount=8000, paid_amount=8000,
        pay_method="test", status=2, expire_time=datetime.now()))
    db.add(SysOrder(
        order_no="test_stat_v2", user_id=USER_ID, package_id=vip.id,
        package_level="level_1", package_name=vip.name, package_type="vip",
        original_price=8000, payable_amount=8000, paid_amount=8000,
        pay_method="test", status=3, expire_time=datetime.now()))
    db.add(SysOrder(
        order_no="test_stat_c1", user_id=USER_ID, package_id=credit.id,
        package_level="", package_name=credit.name, package_type="credit",
        original_price=4000, payable_amount=4000, paid_amount=4000,
        pay_method="test", status=2, expire_time=datetime.now()))
    await db.flush()
    stats = await package_service.get_sales_stats(db)
    type_stats = {t["packageType"]: t for t in stats["typeStats"]}
    assert type_stats["vip"]["packageTypeName"] == "会员卡"
    assert type_stats["vip"]["salesCount"] == 2
    assert type_stats["vip"]["revenue"] == 16000
    assert type_stats["credit"]["packageTypeName"] == "积分卡"
    assert type_stats["credit"]["salesCount"] == 1
    assert type_stats["credit"]["revenue"] == 4000
    # 等级/周期维度仅统计会员卡订单（积分卡无等级与周期）
    level_stats = {i["levelCode"]: i for i in stats["levelStats"]}
    assert set(level_stats) == {"level_1"}
    assert level_stats["level_1"]["salesCount"] == 2
    period_stats = {p["period"]: p["salesCount"] for p in stats["periodStats"]}
    assert period_stats == {"monthly": 2}
