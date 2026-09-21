"""套餐管理路由层测试（权限装饰器 / 参数校验 / 读接口登录态）。

遵循 05-python-test-rules：marker=api、dependency_overrides + monkeypatch service、
只断言业务结果。权限口径（API接口.md §3）：
- 写接口：package:add/edit/delete、package:coupon:*、package:promotion:*
- 读接口（page/form/listOnSale/coupons page/promotions page）：登录即可访问
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.router import package, promotion
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.api

PACKAGE_PERMS = ["package:add", "package:edit", "package:delete", "package:sales"]
COUPON_PERMS = [
    "package:coupon:add",
    "package:coupon:distribute",
    "package:coupon:edit",
    "package:coupon:delete",
]
PROMOTION_PERMS = ["package:promotion:add", "package:promotion:edit", "package:promotion:delete"]
ALL_PERMS = PACKAGE_PERMS + COUPON_PERMS + PROMOTION_PERMS


def _admin_ctx():
    return make_user_context(2, username="admin", roles=["ADMIN"], permissions=ALL_PERMS)


def _plain_ctx():
    """普通用户：无任何 package:* 权限"""
    return make_user_context(5, username="user", roles=[], permissions=[])


def _vip_form(**overrides) -> dict:
    form = {
        "name": "路由测试套餐",
        "packageType": "vip",
        "levelCode": "level_1",
        "period": "monthly",
        "periodDays": 30,
        "originalPrice": 10000,
        "salePrice": 8000,
    }
    form.update(overrides)
    return form


def _coupon_form(**overrides) -> dict:
    form = {
        "name": "路由测试券",
        "type": "full_reduction",
        "faceValue": 1000,
        "threshold": 5000,
        "validType": "relative",
        "validDays": 30,
        "totalQty": 100,
        "perUserLimit": 1,
    }
    form.update(overrides)
    return form


def _promotion_form(**overrides) -> dict:
    form = {
        "name": "路由测试活动",
        "type": "full_reduction",
        "startTime": "2026-01-01 00:00:00",
        "endTime": "2026-12-31 23:59:59",
    }
    form.update(overrides)
    return form


@pytest.fixture
async def package_client():
    async def _override_db():
        return object()

    async def _override_user():
        return _admin_ctx()

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as client:
        yield client
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)


async def _as_plain(client: AsyncClient):
    async def _override_user():
        return _plain_ctx()

    fastapi_app.dependency_overrides[get_current_user] = _override_user
    return client


# ===== 套餐写接口越权（A0301） =====


async def test_add_package_without_permission_forbidden(package_client):
    await _as_plain(package_client)
    resp = await package_client.post("/api/v1/packages", json=_vip_form())
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_update_package_without_permission_forbidden(package_client):
    await _as_plain(package_client)
    resp = await package_client.put("/api/v1/packages/1", json=_vip_form())
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_update_package_status_without_permission_forbidden(package_client):
    await _as_plain(package_client)
    resp = await package_client.put("/api/v1/packages/1/status", params={"status": 0})
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_delete_package_without_permission_forbidden(package_client):
    await _as_plain(package_client)
    resp = await package_client.delete("/api/v1/packages/1")
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_sales_stats_without_permission_forbidden(package_client):
    await _as_plain(package_client)
    resp = await package_client.get("/api/v1/packages/sales/stats")
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


# ===== 优惠券写接口越权（A0301） =====


async def test_add_coupon_without_permission_forbidden(package_client):
    await _as_plain(package_client)
    resp = await package_client.post("/api/v1/packages/coupons", json=_coupon_form())
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_batch_distribute_coupon_without_permission_forbidden(package_client):
    await _as_plain(package_client)
    resp = await package_client.post(
        "/api/v1/packages/coupons/batch",
        json={
            "couponId": 1,
            "targetScope": "users",
            "userIds": [5],
        },
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_update_coupon_without_permission_forbidden(package_client):
    await _as_plain(package_client)
    resp = await package_client.put("/api/v1/packages/coupons/1", json=_coupon_form())
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_delete_coupon_without_permission_forbidden(package_client):
    await _as_plain(package_client)
    resp = await package_client.delete("/api/v1/packages/coupons/1")
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


# ===== 促销活动写接口越权（A0301） =====


async def test_add_promotion_without_permission_forbidden(package_client):
    await _as_plain(package_client)
    resp = await package_client.post("/api/v1/packages/promotions", json=_promotion_form())
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_update_promotion_without_permission_forbidden(package_client):
    await _as_plain(package_client)
    resp = await package_client.put("/api/v1/packages/promotions/1", json=_promotion_form())
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_update_promotion_status_without_permission_forbidden(package_client):
    await _as_plain(package_client)
    resp = await package_client.put("/api/v1/packages/promotions/1/status", params={"status": 1})
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_delete_promotion_without_permission_forbidden(package_client):
    await _as_plain(package_client)
    resp = await package_client.delete("/api/v1/packages/promotions/1")
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_bind_promotion_packages_without_permission_forbidden(package_client):
    await _as_plain(package_client)
    resp = await package_client.put(
        "/api/v1/packages/promotions/1/packages", json={"packageIds": [1]}
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


# ===== 参数校验（全局 handler：HTTP 400 + A0400，T-PM-016/024/025/026/027/028） =====


async def test_add_package_invalid_package_type_rejected(package_client):
    resp = await package_client.post("/api/v1/packages", json=_vip_form(packageType="other"))
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_add_package_name_too_short_rejected(package_client):
    resp = await package_client.post("/api/v1/packages", json=_vip_form(name="A"))
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_add_package_name_too_long_rejected(package_client):
    resp = await package_client.post("/api/v1/packages", json=_vip_form(name="名" * 33))
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_add_package_period_days_zero_rejected(package_client):
    resp = await package_client.post("/api/v1/packages", json=_vip_form(periodDays=0))
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_add_package_period_days_over_365_rejected(package_client):
    resp = await package_client.post("/api/v1/packages", json=_vip_form(periodDays=366))
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_add_package_original_price_zero_rejected(package_client):
    resp = await package_client.post(
        "/api/v1/packages", json=_vip_form(originalPrice=0, salePrice=0)
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_add_package_negative_price_rejected(package_client):
    resp = await package_client.post(
        "/api/v1/packages", json=_vip_form(originalPrice=-1, salePrice=-1)
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_add_package_sort_out_of_range_rejected(package_client):
    resp = await package_client.post("/api/v1/packages", json=_vip_form(sort=1000))
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_add_package_description_too_long_rejected(package_client):
    resp = await package_client.post("/api/v1/packages", json=_vip_form(description="描" * 257))
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_add_package_invalid_period_rejected(package_client):
    resp = await package_client.post("/api/v1/packages", json=_vip_form(period="daily"))
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_add_package_benefit_overrides_wrong_type_rejected(package_client):
    # T-PM-029：benefitOverrides 结构非法（字符串）→ 校验失败，不静默降级
    resp = await package_client.post(
        "/api/v1/packages", json=_vip_form(benefitOverrides="{invalid")
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


# ===== 读接口登录即可访问（无 package:* 权限不拦） =====


async def test_read_endpoints_accessible_without_package_permissions(package_client, monkeypatch):
    await _as_plain(package_client)

    async def fake_get_page(db, query):
        return {"list": [], "total": 0}

    async def fake_list_on_sale(db, package_type=None):
        return []

    async def fake_coupon_page(db, query):
        return {"list": [], "total": 0}

    async def fake_promotion_page(db, **kwargs):
        return {"list": [], "total": 0}

    monkeypatch.setattr(package.package_service, "get_page", fake_get_page)
    monkeypatch.setattr(package.package_service, "list_on_sale", fake_list_on_sale)
    monkeypatch.setattr(package.coupon_service, "get_page", fake_coupon_page)
    monkeypatch.setattr(promotion.promotion_service, "get_page", fake_promotion_page)

    for url in [
        "/api/v1/packages/page",
        "/api/v1/packages",
        "/api/v1/packages/coupons/page",
        "/api/v1/packages/promotions/page",
    ]:
        resp = await package_client.get(url)
        assert resp.status_code == 200, url
        assert resp.json()["code"] == "00000", url
