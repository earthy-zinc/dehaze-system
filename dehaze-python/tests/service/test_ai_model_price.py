"""模型用户售价服务单元测试：价格版本化 CRUD、版本递增、三维档位匹配换算积分、未配置返回默认"""

from datetime import datetime
from decimal import Decimal

import pytest

pytestmark = pytest.mark.requires_db

from app.models.schema.ai_model_price import (
    ModelPriceCreateRequest,
    ModelPriceDetailForm,
    ModelPriceQuery,
)
from app.repository.ai_model_price_repository import ai_model_price_repository
from app.service.ai_model_price_service import AiModelPriceService
from app.service.billing.rate_provider import RateProvider

# 2026-08-17 周一（高峰）、2026-08-22 周六（空闲）
_PEAK_TIME = datetime(2026, 8, 17, 10, 0, 0)
_IDLE_TIME = datetime(2026, 8, 22, 10, 0, 0)
# 售价版本生效时间需早于核算调用时刻
_EFFECTIVE_FROM = datetime(2026, 1, 1)


def _create_request(model_id, details):
    return ModelPriceCreateRequest(
        model_id=model_id, provider_id=1, effective_from=_EFFECTIVE_FROM, details=details,
    )


def _detail(token_type, unit_price, time_slot="idle", min_tokens=0, max_tokens=None):
    return ModelPriceDetailForm(
        token_type=token_type,
        time_slot=time_slot,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        unit_price=Decimal(unit_price),
    )


class TestPriceVersioning:
    async def test_create_price_increments_price_version(self, db):
        svc = AiModelPriceService(price_repository=ai_model_price_repository)
        req1 = _create_request("deepseek-v4-flash", [_detail("input", "2000")])
        req2 = _create_request("deepseek-v4-flash", [_detail("input", "3000")])
        p1 = await svc.create_price(db, req1)
        p2 = await svc.create_price(db, req2)

        assert p1.price_version == 1
        assert p2.price_version == 2
        assert p1.details[0].unit_price == Decimal("2000")
        assert p2.details[0].unit_price == Decimal("3000")

    async def test_price_version_not_reused_after_delete(self, db):
        svc = AiModelPriceService(price_repository=ai_model_price_repository)
        p = await svc.create_price(db, _create_request("m", [_detail("input", "2000")]))
        await svc.delete_price(db, p.id)
        p2 = await svc.create_price(db, _create_request("m", [_detail("input", "2000")]))
        # 软删版本号不可复用（联合唯一键），新版本号继续递增
        assert p2.price_version == 2

    async def test_update_and_delete_price(self, db):
        svc = AiModelPriceService(price_repository=ai_model_price_repository)
        created = await svc.create_price(db, _create_request("m", [_detail("input", "2000")]))
        updated = await svc.update_price(db, created.id, {"status": 0})
        assert updated.status == 0

        await svc.delete_price(db, created.id)
        # service 测试环境未注册全局软删过滤事件，显式查含软删行校验 deleted=1
        soft_deleted = await ai_model_price_repository.get_by_id(
            db, created.id, with_deleted=True
        )
        assert soft_deleted is not None
        assert soft_deleted.deleted == 1
        # 主表软删时档位明细一并软删（list_details 走软删过滤，此处直接查含软删行）
        from sqlalchemy import select
        from app.models.entity.sys_ai_model_price import SysAiModelPriceDetail

        stmt = (
            select(SysAiModelPriceDetail)
            .where(SysAiModelPriceDetail.price_id == created.id)
            .execution_options(include_deleted=True)
        )
        details = (await db.execute(stmt)).scalars().all()
        assert details and details[0].deleted == 1

    async def test_list_prices_filters_by_model(self, db):
        svc = AiModelPriceService(price_repository=ai_model_price_repository)
        await svc.create_price(db, _create_request("m-a", []))
        await svc.create_price(db, _create_request("m-b", []))

        result = await svc.list_prices(db, ModelPriceQuery(model_id="m-a", page=1, size=10))
        assert result.total == 1
        assert result.list[0].model_id == "m-a"


class TestCalculateCredits:
    async def test_no_price_config_returns_zero(self, db):
        svc = AiModelPriceService(price_repository=ai_model_price_repository)
        result = await svc.calculate(db, "unknown-model", 1, _IDLE_TIME, 900, 100, 500)
        assert result == {"credits": 0, "credits_saved": 0, "configured": False}

    async def test_idle_tier_matching(self, db):
        svc = AiModelPriceService(price_repository=ai_model_price_repository)
        await svc.create_price(
            db,
            _create_request("m", [
                _detail("input", "2000", min_tokens=0, max_tokens=100),
                _detail("input", "3000", min_tokens=100),
                _detail("cached", "500"),
                _detail("output", "6000"),
            ]),
        )
        # input_tokens 含缓存命中部分：未命中 900-100=800；total_input=1000 → 命中第二段 3000
        # credits = 800×3000/1M + 100×500/1M + 500×6000/1M = 2.4+0.05+3 = 5.45 ≈ 5
        # credits_saved = 100×(3000-500)/1M = 0.25 → 0
        result = await svc.calculate(db, "m", 1, _IDLE_TIME, 900, 100, 500)
        assert result == {"credits": 5, "credits_saved": 0, "configured": True}

    async def test_peak_slot_uses_peak_price(self, db):
        svc = AiModelPriceService(price_repository=ai_model_price_repository)
        await svc.create_price(
            db,
            _create_request("m", [
                _detail("input", "2000", time_slot="idle"),
                _detail("input", "4000", time_slot="peak"),
            ]),
        )
        idle_result = await svc.calculate(db, "m", 1, _IDLE_TIME, 1000, 0, 0)
        peak_result = await svc.calculate(db, "m", 1, _PEAK_TIME, 1000, 0, 0)
        assert idle_result["credits"] == 2  # 1000×2000/1M
        assert peak_result["credits"] == 4  # 1000×4000/1M

    async def test_rounding_up_and_min_one_credit(self, db):
        svc = AiModelPriceService(price_repository=ai_model_price_repository)
        await svc.create_price(
            db,
            _create_request("m", [
                _detail("input", "100"),
                _detail("output", "100"),
            ]),
        )
        # credits = 1000×100/1M + 500×100/1M = 0.15 → ROUND_HALF_UP → 0 → 至少 1 积分
        result = await svc.calculate(db, "m", 1, _IDLE_TIME, 1000, 0, 500)
        assert result["credits"] == 1


class TestPriceBoundary:
    async def test_unit_price_zero_accepted(self, db):
        svc = AiModelPriceService(price_repository=ai_model_price_repository)
        req = _create_request("zero-price", [_detail("input", "0")])
        p = await svc.create_price(db, req)
        assert p.details[0].unit_price == Decimal("0")
        result = await svc.calculate(db, "zero-price", 1, _IDLE_TIME, 1000, 0, 0)
        assert result["credits"] == 0
        assert result["configured"] is True

    async def test_unit_price_negative_rejected(self, db):
        svc = AiModelPriceService(price_repository=ai_model_price_repository)
        with pytest.raises(Exception):
            _create_request("neg-price", [_detail("input", "-1")])

    async def test_unit_price_max_value_accepted(self, db):
        svc = AiModelPriceService(price_repository=ai_model_price_repository)
        req = _create_request("max-price", [_detail("input", "99999999.9999")])
        p = await svc.create_price(db, req)
        assert p.details[0].unit_price == Decimal("99999999.9999")
        result = await svc.calculate(db, "max-price", 1, _IDLE_TIME, 1000, 0, 0)
        assert result["credits"] == 100000  # 1000 * 99999999.9999 / 1e6 ≈ 99999.9999 → 100000


class TestExternalPaidModels:
    """外部付费模型（gpt-4o-mini / deepseek-v4-flash）售价换算单测。

    价格配置对齐开发库迁移恢复的原费率（gpt-4o-mini input=1.0/output=4.0/cached=0.5；
    deepseek-v4-flash input=1.0/output=3.0/cached=0.5，每token积分 = 1M/4M(3M)/500k 每百万token）。
    仅本地 DB 档位换算，不调用真实付费 API，零费用、常跑。
    """

    async def test_gpt4o_mini_credits_and_saved(self, db):
        svc = AiModelPriceService(price_repository=ai_model_price_repository)
        await svc.create_price(
            db,
            _create_request("gpt-4o-mini", [
                _detail("input", "1000000"),
                _detail("cached", "500000"),
                _detail("output", "4000000"),
            ]),
        )
        # input=1000 含缓存 200：未命中 800×1M/1M + 缓存 200×500k/1M + 输出 300×4M/1M = 2100
        # credits_saved = 200×(1M-500k)/1M = 100
        result = await svc.calculate(db, "gpt-4o-mini", 1, _IDLE_TIME, 1000, 200, 300)
        assert result == {"credits": 2100, "credits_saved": 100, "configured": True}

    async def test_deepseek_v4_flash_peak_vs_idle(self, db):
        svc = AiModelPriceService(price_repository=ai_model_price_repository)
        await svc.create_price(
            db,
            _create_request("deepseek-v4-flash", [
                _detail("input", "1000000", time_slot="idle"),
                _detail("input", "1500000", time_slot="peak"),
                _detail("cached", "100000", time_slot="idle"),
                _detail("cached", "100000", time_slot="peak"),
                _detail("output", "3000000", time_slot="idle"),
                _detail("output", "3000000", time_slot="peak"),
            ]),
        )
        # peak: 1000×1.5M/1M + 500×3M/1M = 3000；idle: 1000×1M/1M + 500×3M/1M = 2500
        peak = await svc.calculate(db, "deepseek-v4-flash", 1, _PEAK_TIME, 1000, 0, 500)
        idle = await svc.calculate(db, "deepseek-v4-flash", 1, _IDLE_TIME, 1000, 0, 500)
        assert peak["credits"] == 3000
        assert idle["credits"] == 2500

    async def test_rate_provider_paid_model_calculate(self, db):
        """付费模型经 rate_provider.calculate 走真实价格版本换算（非 mock），结算闭环"""
        svc = AiModelPriceService(price_repository=ai_model_price_repository)
        await svc.create_price(
            db,
            _create_request("gpt-4o-mini", [
                _detail("input", "1000000", time_slot="idle"),
                _detail("input", "1000000", time_slot="peak"),
                _detail("output", "4000000", time_slot="idle"),
                _detail("output", "4000000", time_slot="peak"),
            ]),
        )
        provider = RateProvider(
            ai_model_price_repository=ai_model_price_repository,
            ai_model_price_service=svc,
        )
        # calculate 内部 at_time=当前时刻，双时段档位同价保证任意时段命中
        # (input=1000, output=200, cached=300)：未命中 700×1M/1M + 200×4M/1M = 1500
        calc = await provider.calculate(db, "gpt-4o-mini", 1, 1000, 200, 300)
        assert calc["credits"] == 1500
        assert calc["configured"] is True
