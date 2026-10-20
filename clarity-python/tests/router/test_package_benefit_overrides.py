"""套餐 benefitOverrides 字段完整性（保存套餐不得静默丢 key）。

回归：`BenefitOverrides` 曾只声明 9 字段，而读侧 `BENEFIT_FIELDS` 是 17 字段全集；
pydantic 默认 `extra="ignore"` + `model_dump(exclude_none=True)` 落库，导致后台保存
套餐时 DB 里已存在的另外 8 个 key 被静默抹掉（数据丢失，非报错）。
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.models.schema.package import BenefitOverrides, PackageForm
from app.router import package as package_module
from app.service.package_service import BENEFIT_FIELDS
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.api

_ALL_BENEFIT_KEYS = {
    "monthlyDehazeQuota",
    "monthlyDerainQuota",
    "monthlyDesnowQuota",
    "monthlyLowlightQuota",
    "monthlySuperResolutionQuota",
    "monthlyDenoiseQuota",
    "monthlyInpaintQuota",
    "monthlyEvaluateQuota",
    "aiCreditsDaily",
    "aiCreditsMonthly",
    "historyRetention",
    "batchLimit",
    "priority",
    "advancedParams",
    "hdExport",
    "reportExport",
    "batchDownload",
}


def test_override_model_covers_full_benefit_field_set():
    """契约字段集必须与读侧 BENEFIT_FIELDS 全集逐一对应（漏声明即静默丢 key）。"""
    assert set(BenefitOverrides.model_fields) == set(BENEFIT_FIELDS)
    assert len(BENEFIT_FIELDS) == 17


def test_form_roundtrip_keeps_every_override_key():
    """17 key 表单 → model_dump(exclude_none=True) → 17 key 仍在。"""
    form = PackageForm(
        name="测试套餐",
        originalPrice=100,
        salePrice=50,
        benefitOverrides=BenefitOverrides(**dict.fromkeys(_ALL_BENEFIT_KEYS, 7)),
    )

    dumped = form.model_dump(exclude_none=True)

    assert set(dumped["benefitOverrides"]) == _ALL_BENEFIT_KEYS


@pytest.fixture
async def client():
    async def _override_db():
        return object()

    async def _override_user():
        return make_user_context(1, username="admin", permissions=["package:edit"])

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    async with AsyncClient(transport=ASGITransport(app=fastapi_app), base_url="http://test") as c:
        yield c
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)


async def test_update_forwards_full_override_set(client, monkeypatch):
    """后台保存套餐时，17 个 override key 必须完整到达 service（不被 schema 过滤）。"""
    captured = {}

    async def _fake_update(db, package_id, data):
        captured.update(data)
        return

    monkeypatch.setattr(package_module.package_service, "update", _fake_update)
    resp = await client.put(
        "/api/v1/packages/1",
        json={
            "name": "测试套餐",
            "originalPrice": 100,
            "salePrice": 50,
            "benefitOverrides": dict.fromkeys(_ALL_BENEFIT_KEYS, 3),
        },
    )

    assert resp.status_code == 200
    assert set(captured["benefitOverrides"]) == _ALL_BENEFIT_KEYS
