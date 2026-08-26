"""成本管理服务单元测试：价格版本化、档位匹配核算、成本回填、双口径统计、对账"""

from datetime import datetime
from decimal import Decimal

import pytest

pytestmark = pytest.mark.requires_db

from app.models.entity.sys_ai_billing import SysAiBilling
from app.models.entity.sys_order import SysOrder
from app.models.schema.ai_billing_cost import (
    ModelCostCreateRequest,
    ModelCostDetailForm,
    ModelCostQuery,
)
from app.repository.ai_billing_repository import ai_billing_repository
from app.repository.ai_model_cost_repository import ai_model_cost_repository
from app.service.billing.cost_service import CostService
from app.service.billing.cost_stat_service import CostStatService

# 2026-08-17 周一（高峰）、2026-08-22 周六（空闲）
_PEAK_TIME = datetime(2026, 8, 17, 10, 0, 0)
_IDLE_TIME = datetime(2026, 8, 22, 10, 0, 0)
# 成本版本生效时间需早于核算调用时刻
_EFFECTIVE_FROM = datetime(2026, 1, 1)


def _create_request(model_id, details):
    return ModelCostCreateRequest(
        model_id=model_id, provider_id=1, effective_from=_EFFECTIVE_FROM, details=details,
    )


def _detail(token_type, unit_price, time_slot="idle", min_tokens=0, max_tokens=None):
    return ModelCostDetailForm(
        token_type=token_type,
        time_slot=time_slot,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        unit_price=Decimal(unit_price),
    )


class TestCostVersioning:
    async def test_create_cost_increments_price_version(self, db):
        svc = CostService(ai_model_cost_repository=ai_model_cost_repository)
        req1 = _create_request("deepseek-v4-flash", [_detail("input", "2")])
        req2 = _create_request("deepseek-v4-flash", [_detail("input", "3")])
        c1 = await svc.create_cost(db, req1)
        c2 = await svc.create_cost(db, req2)

        assert c1.price_version == 1
        assert c2.price_version == 2
        assert c1.details[0].unit_price == Decimal("2")
        assert c2.details[0].unit_price == Decimal("3")

    async def test_update_and_delete_cost(self, db):
        svc = CostService(ai_model_cost_repository=ai_model_cost_repository)
        created = await svc.create_cost(db, _create_request("m", [_detail("input", "2")]))
        updated = await svc.update_cost(db, created.id, {"status": 0})
        assert updated.status == 0

        await svc.delete_cost(db, created.id)
        # service 测试环境未注册全局软删过滤事件，显式查含软删行校验 deleted=1
        soft_deleted = await ai_model_cost_repository.get_by_id(
            db, created.id, with_deleted=True
        )
        assert soft_deleted is not None
        assert soft_deleted.deleted == 1

    async def test_list_costs_filters_by_model(self, db):
        svc = CostService(ai_model_cost_repository=ai_model_cost_repository)
        await svc.create_cost(db, _create_request("m-a", []))
        await svc.create_cost(db, _create_request("m-b", []))

        result = await svc.list_costs(db, ModelCostQuery(model_id="m-a", page=1, size=10))
        assert result.total == 1
        assert result.list[0].model_id == "m-a"


class TestCalculateCost:
    async def test_no_cost_config_returns_zero(self, db):
        svc = CostService(ai_model_cost_repository=ai_model_cost_repository)
        cost = await svc.calculate_cost(db, "unknown-model", 1, _IDLE_TIME, 900, 100, 500)
        assert cost == Decimal("0")

    async def test_idle_tier_matching(self, db):
        svc = CostService(ai_model_cost_repository=ai_model_cost_repository)
        await svc.create_cost(
            db,
            _create_request("m", [
                _detail("input", "2", min_tokens=0, max_tokens=100),
                _detail("input", "3", min_tokens=100),
                _detail("cached", "0.5"),
                _detail("output", "6"),
            ]),
        )
        # input_tokens 含缓存命中部分：未命中 900-100=800；total_input=1000 → 命中第二段 3
        # cost = 800×3 + 100×0.5 + 500×6 = 5450 → 0.00545 → ROUND_HALF_EVEN → 0.0054
        cost = await svc.calculate_cost(db, "m", 1, _IDLE_TIME, 900, 100, 500)
        assert cost == Decimal("0.0054")

    async def test_peak_slot_uses_peak_price(self, db):
        svc = CostService(ai_model_cost_repository=ai_model_cost_repository)
        await svc.create_cost(
            db,
            _create_request("m", [
                _detail("input", "2", time_slot="idle"),
                _detail("input", "4", time_slot="peak"),
            ]),
        )
        idle_cost = await svc.calculate_cost(db, "m", 1, _IDLE_TIME, 1000, 0, 0)
        peak_cost = await svc.calculate_cost(db, "m", 1, _PEAK_TIME, 1000, 0, 0)
        assert idle_cost == Decimal("0.0020")
        assert peak_cost == Decimal("0.0040")

    async def test_backfill_cost_updates_billing(self, db):
        svc = CostService(
            ai_model_cost_repository=ai_model_cost_repository,
            ai_billing_repository=ai_billing_repository,
        )
        # 同时配置 peak/idle 档位：billing.create_time 为当前时刻，时段判定不确定
        await svc.create_cost(
            db,
            _create_request("m", [
                _detail("input", "2", time_slot="idle"),
                _detail("input", "2", time_slot="peak"),
                _detail("output", "6", time_slot="idle"),
                _detail("output", "6", time_slot="peak"),
            ]),
        )
        billing = SysAiBilling(
            user_id=1, model="m", provider_id=1, bill_type="chat",
            input_tokens=1000, cached_input_tokens=0, output_tokens=500,
            credits=100, quota_consumed=100, pre_deduct=100,
        )
        db.add(billing)
        await db.flush()

        await svc.backfill_cost(db, billing.id)
        await db.refresh(billing)
        assert billing.cost == Decimal("0.0050")  # 1000×2 + 500×6 = 5000/1e6


class TestCostStats:
    async def test_dual_metric_gross_profit(self, db):
        now = datetime.now()
        db.add(SysAiBilling(
            user_id=1, model="m", bill_type="chat",
            input_tokens=10, output_tokens=0, credits=100, quota_consumed=100, pre_deduct=100,
            cost=Decimal("0.5000"),
        ))
        db.add(SysOrder(
            order_no="T-CREDIT-1", user_id=1, package_id=1, package_name="积分卡",
            package_type="credit", package_level=None, original_price=100,
            payable_amount=100, paid_amount=100, status=3, expire_time=now, paid_time=now,
        ))
        db.add(SysOrder(
            order_no="T-VIP-1", user_id=1, package_id=2, package_name="会员卡",
            package_type="vip", package_level="level_1", original_price=200,
            payable_amount=200, paid_amount=200, status=2, expire_time=now, paid_time=now,
        ))
        await db.flush()

        svc = CostStatService()
        stats = await svc.cost_stats(db)
        by_metric = {s.metric: s for s in stats}
        assert set(by_metric) == {"overall", "ai"}

        overall = by_metric["overall"]
        assert overall.revenue == 3.0  # (100+200)/100 分转元
        assert overall.cost == 0.5
        assert overall.profit == 2.5

        ai = by_metric["ai"]
        assert ai.revenue == 1.6  # 1 + 2×0.3
        assert ai.profit == 1.1


class TestImportReconcile:
    def test_counts_non_empty_lines(self):
        assert CostService.import_reconcile("a\nb\n\nc") == 3
        assert CostService.import_reconcile("") == 0
        assert CostService.import_reconcile("单行") == 1
