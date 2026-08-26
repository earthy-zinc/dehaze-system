"""
促销活动 service 单元测试（真实 MySQL 测试库 dehaze_test + SAVEPOINT 回滚）。

覆盖 promotion_service 核心逻辑：
create / update / update_status（缓存失效）/ bind_packages（先删后插、折扣取自活动 activity_rules）
/ delete（逻辑删除+删关联）/ get_page（分页与筛选）/ list_active_by_package_id。

缓存失效通过 mock_redis 断言 del 被调用验证。
"""

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_promotion import SysPromotionPackage
from app.models.schema.promotion import PromotionForm, PromotionPackageForm
from app.repository.promotion_repository import promotion_repository
from app.service.promotion_service import promotion_service

pytestmark = pytest.mark.requires_db

USER_ID = 1000002


def _promo_payload(name: str, start, end, extra: dict | None = None, status: int | None = 1, promo_type: str = "full_reduction") -> dict:
    base = {
        "name": name,
        "type": promo_type,
        "startTime": start,
        "endTime": end,
        "activityRules": {"discount_type": "percent", "discount_value": 10},
        "newUserOnly": 0,
    }
    if status is not None:
        base["status"] = status
    if extra:
        base.update(extra)
    return base


async def _make_promo(db, name: str, discount_type="percent", discount_value=10, status: int | None = 1, promo_type: str = "full_reduction"):
    from datetime import datetime, timedelta
    now = datetime.now()
    promo = await promotion_service.create(db, PromotionForm(**_promo_payload(
        name,
        (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
        (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
        extra={"activityRules": {"discount_type": discount_type, "discount_value": discount_value}},
        status=status,
        promo_type=promo_type,
    )))
    return promo


async def test_create_default_status(db):
    promo = await _make_promo(db, "test_promo_c1", status=None)
    assert promo["id"] is not None
    assert promo["status"] == 0  # 缺省 0


async def test_update(db):
    promo = await _make_promo(db, "test_promo_u1")
    await promotion_service.update(db, promo["id"], PromotionForm(
        name="test_promo_u1_new",
        type="full_reduction",
        startTime=promo["startTime"],
        endTime=promo["endTime"],
        activityRules={"discount_type": "percent", "discount_value": 10},
        newUserOnly=0,
    ))
    updated = await promotion_repository.get_by_id(db, promo["id"])
    assert updated.name == "test_promo_u1_new"


async def test_update_status_invalidates_cache(db, mock_redis):
    from app.models.entity.sys_package import SysPackage
    from app.repository.package_repository import package_repository
    pkg = SysPackage(
        name="test_pkg_cache", package_type="credit", credit_amount=1000,
        original_price=5000, sale_price=4000, status=0,
    )
    await package_repository.create(db, pkg)
    promo = await _make_promo(db, "test_promo_uc")
    await promotion_service.bind_packages(
        db, promo["id"], PromotionPackageForm(packageIds=[pkg.id]))
    # 预热绑定套餐的详情缓存与在售缓存（键名与 promotion/package 对齐）
    await mock_redis.set(f"package:detail:{pkg.id}", "x")
    await mock_redis.set("package:onsale:credit", "y")
    await promotion_service.update_status(db, promo["id"], 0)
    assert await mock_redis.get(f"package:detail:{pkg.id}") is None
    assert await mock_redis.get("package:onsale:credit") is None


async def test_bind_packages_replace_old(db):
    promo = await _make_promo(db, "test_promo_bp", discount_type="percent", discount_value=10)
    await promotion_service.bind_packages(db, promo["id"], PromotionPackageForm(packageIds=[101, 102]))
    # 重新绑定，应替换旧关联
    await promotion_service.bind_packages(db, promo["id"], PromotionPackageForm(packageIds=[201]))
    ids = await promotion_repository.list_package_ids_by_promotion(db, promo["id"])
    assert ids == [201]


async def test_bind_packages_default_discount(db):
    # 绑定折扣取自活动 activity_rules 的 discount_type/discount_value
    promo = await _make_promo(db, "test_promo_bd", discount_type="percent", discount_value=0)
    await promotion_service.bind_packages(db, promo["id"], PromotionPackageForm(packageIds=[301]))
    rows = await promotion_repository.list_active_by_package_id(db, 301)
    assert len(rows) == 1
    assert rows[0]["promotion_package"].discount_type == "percent"


async def test_delete_logical(db):
    promo = await _make_promo(db, "test_promo_dl", discount_type="percent", discount_value=10)
    await promotion_service.bind_packages(db, promo["id"], PromotionPackageForm(packageIds=[401]))
    await promotion_service.delete(db, promo["id"])
    detail = await promotion_repository.get_by_id(db, promo["id"])
    assert detail.deleted == 1
    ids = await promotion_repository.list_package_ids_by_promotion(db, promo["id"])
    assert ids == []


async def test_get_page_filter(db):
    await _make_promo(db, "test_promo_pg1", discount_type="percent", discount_value=10, promo_type="full_reduction")
    await _make_promo(db, "test_promo_pg2", discount_type="fixed", discount_value=500, promo_type="discount")
    res = await promotion_service.get_page(db, 1, 10, type="discount")
    assert res["total"] >= 1
    assert all(p["type"] == "discount" for p in res["list"])
    res2 = await promotion_service.get_page(db, 1, 10, status=1)
    assert all(p["status"] == 1 for p in res2["list"])


async def test_list_active_by_package_id(db):
    promo = await _make_promo(db, "test_promo_la", discount_type="percent", discount_value=20)
    await promotion_service.bind_packages(db, promo["id"], PromotionPackageForm(packageIds=[501]))
    active = await promotion_repository.list_active_by_package_id(db, 501)
    assert len(active) == 1
    assert active[0]["promotion"].id == promo["id"]
    assert await promotion_repository.list_active_by_package_id(db, 999) == []
