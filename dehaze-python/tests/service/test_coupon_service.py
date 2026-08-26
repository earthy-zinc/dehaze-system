"""
优惠券 service 单元测试（真实 MySQL 测试库 dehaze_test + SAVEPOINT 回滚）。

覆盖 coupon_service 核心逻辑：
create（满减缺门槛/固定有效期缺起止时间/正常创建含商品限定）
/ receive（正常领取/超库存/超每人限领/trial 券限领 1 次/无限量券）
/ batch_distribute（users 发放，返回成功失败计数）
/ list_my（按状态筛选）。
"""

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_coupon import SysCoupon
from app.repository.coupon_repository import coupon_repository
from app.service.coupon_service import coupon_service

pytestmark = pytest.mark.requires_db

USER_ID = 1000003


def _coupon_payload(name: str, extra: dict | None = None) -> dict:
    base = {
        "name": name,
        "type": "no_threshold",
        "faceValue": 1000,
        "totalQty": 100,
        "perUserLimit": 1,
        "validType": "permanent",
        "applicableScope": [],
        "status": 1,
    }
    if extra:
        base.update(extra)
    return base


async def _create_coupon(db, payload: dict) -> SysCoupon:
    result = await coupon_service.create(db, payload)
    return await coupon_repository.get_by_id(db, result["id"])


async def test_create_full_reduction_missing_threshold(db):
    with pytest.raises(BusinessException) as exc:
        await coupon_service.create(db, _coupon_payload(
            "测试券-满减缺门槛", extra={
                "type": "full_reduction", "discountValue": 0})
        )
    assert exc.value.code == ResultCode.BUSINESS_ERROR
    assert "门槛" in exc.value.message


async def test_create_fixed_missing_time(db):
    with pytest.raises(BusinessException) as exc:
        await coupon_service.create(db, _coupon_payload(
            "测试券-固定缺时间", extra={
                "validType": "fixed", "validStart": None, "validEnd": None})
        )
    assert exc.value.code == ResultCode.BUSINESS_ERROR
    assert "起止时间" in exc.value.message


async def test_create_with_applicable_scope(db):
    coupon = await _create_coupon(db, _coupon_payload(
        "测试券-限定商品", extra={"applicableScope": [123]})
    )
    assert coupon.id is not None
    assert coupon.applicable_scope == [123]


async def test_receive_success(db):
    coupon = await _create_coupon(db, _coupon_payload("测试券-领取"))
    result = await coupon_service.receive(db, coupon.id, USER_ID)
    assert result["userCouponId"] is not None


async def test_receive_stock_empty(db):
    coupon = await _create_coupon(db, _coupon_payload(
        "测试券-无库存", extra={"totalQty": 0}))
    with pytest.raises(BusinessException) as exc:
        await coupon_service.receive(db, coupon.id, USER_ID)
    assert exc.value.code == ResultCode.COUPON_STOCK_EMPTY


async def test_receive_limit_exceeded(db):
    coupon = await _create_coupon(db, _coupon_payload(
        "测试券-限领", extra={"totalQty": 100, "perUserLimit": 1}))
    await coupon_service.receive(db, coupon.id, USER_ID)
    with pytest.raises(BusinessException) as exc:
        await coupon_service.receive(db, coupon.id, USER_ID)
    assert exc.value.code == ResultCode.COUPON_LIMIT_EXCEEDED


async def test_receive_trial_once(db):
    coupon = await _create_coupon(db, _coupon_payload(
        "测试券-体验", extra={"type": "trial", "faceValue": 0, "perUserLimit": 1}))
    await coupon_service.receive(db, coupon.id, USER_ID)
    with pytest.raises(BusinessException) as exc:
        await coupon_service.receive(db, coupon.id, USER_ID)
    assert "体验券每人限领 1 次" in exc.value.message


async def test_receive_unlimited(db):
    coupon = await _create_coupon(db, _coupon_payload(
        "测试券-无限", extra={"totalQty": -1, "perUserLimit": 5}))
    r1 = await coupon_service.receive(db, coupon.id, USER_ID)
    r2 = await coupon_service.receive(db, coupon.id, USER_ID + 1)
    assert r1["userCouponId"] is not None
    assert r2["userCouponId"] is not None


async def test_batch_distribute_users(db):
    coupon = await _create_coupon(db, _coupon_payload(
        "测试券-指定", extra={"totalQty": -1, "perUserLimit": 5}))
    result = await coupon_service.batch_distribute(db, {
        "couponId": coupon.id,
        "targetScope": "users",
        "userIds": [USER_ID, USER_ID + 1],
    })
    assert result["successCount"] == 2
    assert result["failCount"] == 0


async def test_batch_distribute_all(db):
    coupon = await _create_coupon(db, _coupon_payload(
        "测试券-全量", extra={"totalQty": -1, "perUserLimit": 5}))
    result = await coupon_service.batch_distribute(db, {
        "couponId": coupon.id,
        "targetScope": "all",
    })
    assert "successCount" in result
    assert "failCount" in result
    assert result["successCount"] >= 1


async def test_list_my_by_status(db):
    coupon = await _create_coupon(db, _coupon_payload(
        "测试券-我的", extra={"totalQty": -1, "perUserLimit": 5}))
    await coupon_service.receive(db, coupon.id, USER_ID)
    unused = await coupon_service.list_my(db, USER_ID, status=1)
    assert any(uc["couponId"] == coupon.id for uc in unused)
    used = await coupon_service.list_my(db, USER_ID, status=2)
    assert all(uc["couponId"] != coupon.id for uc in used)
