"""AI 计费管理路由测试：userId 下钻查询与权限校验"""

import time
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.models.schema.ai_billing import BalanceResult
from app.models.schema.common import PageResult
from app.router import ai_billing as billing_module
from app.service.billing.billing_record_service import billing_record_service
from app.service.billing.refund_service import refund_service

pytestmark = pytest.mark.api


class _FakeUser:
    def __init__(self, id=1, is_root=False, permissions=()):
        self.id = id
        self.is_root = is_root
        self.permissions = list(permissions)


@pytest.fixture
async def ai_billing_client():
    async def _override_db():
        return object()

    current_user = {"user": _FakeUser()}

    async def _override_user():
        return current_user["user"]

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as client:
        yield client, current_user
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)


async def _fake_balance(db, user_id):
    return BalanceResult(
        user_id=user_id,
        credits_balance=Decimal("100"),
        arrears_status=False,
        daily_used=0,
        daily_limit=10000,
        monthly_used=0,
        monthly_limit=100000,
    )


async def _fake_page(db, user_id, query):
    captured["user_id"] = user_id
    return PageResult(list=[], total=0)


captured: dict = {}


def test_billing_query_paths_registered(app):
    schema = app.openapi()
    for path in (
        "/api/v1/ai-billing/balance",
        "/api/v1/ai-billing/records",
        "/api/v1/ai-billing/credit-logs",
        "/api/v1/ai-billing/refunds",
    ):
        assert path in schema["paths"], f"缺少路径 {path}"


class TestBalanceUserId:
    async def test_own_balance_without_user_id(self, ai_billing_client, monkeypatch):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1)
        monkeypatch.setattr(billing_module, "_build_balance", _fake_balance)

        resp = await client.get("/api/v1/ai-billing/balance")
        assert resp.status_code == 200
        assert resp.json()["data"]["userId"] == 1

    async def test_other_user_balance_forbidden(self, ai_billing_client, monkeypatch):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=[])
        monkeypatch.setattr(billing_module, "_build_balance", _fake_balance)

        resp = await client.get("/api/v1/ai-billing/balance", params={"userId": 2})
        assert resp.status_code == 403
        assert resp.json()["code"] == "A0301"

    async def test_other_user_balance_with_stat_permission(self, ai_billing_client, monkeypatch):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=["ai:billing:stat"])
        monkeypatch.setattr(billing_module, "_build_balance", _fake_balance)

        resp = await client.get("/api/v1/ai-billing/balance", params={"userId": 2})
        assert resp.status_code == 200
        assert resp.json()["data"]["userId"] == 2

    async def test_other_user_balance_root(self, ai_billing_client, monkeypatch):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, is_root=True)
        monkeypatch.setattr(billing_module, "_build_balance", _fake_balance)

        resp = await client.get("/api/v1/ai-billing/balance", params={"userId": 2})
        assert resp.status_code == 200
        assert resp.json()["data"]["userId"] == 2

    async def test_balance_invalid_user_id(self, ai_billing_client, monkeypatch):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, is_root=True)
        monkeypatch.setattr(billing_module, "_build_balance", _fake_balance)

        resp = await client.get("/api/v1/ai-billing/balance", params={"userId": 0})
        # 项目自定义 RequestValidationError handler 统一返回 400（非 FastAPI 默认 422）
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"


class TestRecordsUserId:
    async def test_own_records_without_user_id(self, ai_billing_client, monkeypatch):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1)
        monkeypatch.setattr(billing_record_service, "list_by_user", _fake_page)

        resp = await client.get("/api/v1/ai-billing/records")
        assert resp.status_code == 200
        assert captured["user_id"] == 1

    async def test_other_user_records_forbidden(self, ai_billing_client, monkeypatch):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=[])
        monkeypatch.setattr(billing_record_service, "list_by_user", _fake_page)

        resp = await client.get("/api/v1/ai-billing/records", params={"userId": 2})
        assert resp.status_code == 403
        assert resp.json()["code"] == "A0301"

    async def test_other_user_records_with_stat_permission(self, ai_billing_client, monkeypatch):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=["ai:billing:stat"])
        monkeypatch.setattr(billing_record_service, "list_by_user", _fake_page)

        resp = await client.get("/api/v1/ai-billing/records", params={"userId": 2})
        assert resp.status_code == 200
        assert captured["user_id"] == 2


class TestRefundsList:
    async def test_refunds_forbidden_without_permission(self, ai_billing_client):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=[])

        resp = await client.get("/api/v1/ai-billing/refunds")
        assert resp.status_code == 403
        assert resp.json()["code"] == "A0301"

    async def test_refunds_with_permission_reaches_service(self, ai_billing_client, monkeypatch):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=["ai:billing:refund"])

        received = {}

        async def fake_list(db, query):
            received["query"] = query
            return PageResult(list=[], total=0)

        monkeypatch.setattr(refund_service, "list_refunds", fake_list)

        resp = await client.get(
            "/api/v1/ai-billing/refunds",
            params={"userId": 2, "status": 1, "pageNum": 1, "pageSize": 10},
        )
        assert resp.status_code == 200
        assert received["query"].status == 1
        assert received["query"].user_id == 2
        assert received["query"].page == 1
        assert received["query"].size == 10

    async def test_refunds_invalid_status(self, ai_billing_client):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=["ai:billing:refund"])

        resp = await client.get("/api/v1/ai-billing/refunds", params={"status": 4})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"


class TestCreditLogsUserId:
    async def test_own_logs_without_user_id(self, ai_billing_client, monkeypatch):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1)
        monkeypatch.setattr(billing_record_service, "list_credit_logs", _fake_page)

        resp = await client.get("/api/v1/ai-billing/credit-logs")
        assert resp.status_code == 200
        assert captured["user_id"] == 1

    async def test_other_user_logs_forbidden(self, ai_billing_client, monkeypatch):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=[])
        monkeypatch.setattr(billing_record_service, "list_credit_logs", _fake_page)

        resp = await client.get("/api/v1/ai-billing/credit-logs", params={"userId": 2})
        assert resp.status_code == 403
        assert resp.json()["code"] == "A0301"

    async def test_other_user_logs_with_stat_permission(self, ai_billing_client, monkeypatch):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=["ai:billing:stat"])
        monkeypatch.setattr(billing_record_service, "list_credit_logs", _fake_page)

        resp = await client.get("/api/v1/ai-billing/credit-logs", params={"userId": 2})
        assert resp.status_code == 200
        assert captured["user_id"] == 2


class TestAdjustEndpoint:
    """管理员手动调整（T-AB-007）：amount=0 拒绝、越权拒绝、脏语料原因原样透传"""

    async def test_adjust_zero_amount_rejected(self, ai_billing_client):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=["ai:billing:adjust"])

        resp = await client.post(
            "/api/v1/ai-billing/adjust",
            json={"userId": 2, "amount": 0, "reason": "补扣"},
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_adjust_forbidden_without_permission(self, ai_billing_client):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=[])

        resp = await client.post(
            "/api/v1/ai-billing/adjust",
            json={"userId": 2, "amount": 100, "reason": "越权调整"},
        )
        assert resp.status_code == 403
        assert resp.json()["code"] == "A0301"

    async def test_adjust_dirty_reason_forwarded_to_service(self, ai_billing_client, monkeypatch):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=["ai:billing:adjust"])

        # 合法码点 emoji + 零宽字符 + CRLF + 全半角混杂
        dirty = "补扣\r\n零宽​全角ＡＢｃ\U0001f600"
        received = {}

        async def _fake_recharge(db, user_id, amount, source=None, reason=None, operator_id=None):
            received.update(
                user_id=user_id,
                amount=amount,
                source=source,
                reason=reason,
                operator_id=operator_id,
            )
            return Decimal("100")

        monkeypatch.setattr(billing_module.recharge_service, "recharge", _fake_recharge)
        monkeypatch.setattr(billing_module, "_build_balance", _fake_balance)
        monkeypatch.setattr(
            billing_module.user_repository,
            "get_by_id",
            AsyncMock(return_value=object()),
        )

        resp = await client.post(
            "/api/v1/ai-billing/adjust",
            json={"userId": 2, "amount": 50, "reason": dirty},
        )
        assert resp.status_code == 200
        assert received["reason"] == dirty
        assert received["source"] == "admin_adjust"
        assert received["operator_id"] == 1
        assert received["user_id"] == 2

    async def test_adjust_ghost_user_rejected(self, ai_billing_client, monkeypatch):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=["ai:billing:adjust"])

        monkeypatch.setattr(
            billing_module.user_repository, "get_by_id", AsyncMock(return_value=None)
        )
        recharge_mock = AsyncMock(return_value=Decimal("100"))
        monkeypatch.setattr(billing_module.recharge_service, "recharge", recharge_mock)

        resp = await client.post(
            "/api/v1/ai-billing/adjust",
            json={"userId": 99999999, "amount": 100, "reason": "幽灵充值"},
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0401"
        recharge_mock.assert_not_called()


class TestAdminPermissionMatrix:
    """T-AB-044f/T-AB-050：异常清单与成本接口仅管理员可查"""

    async def test_stats_forbidden_without_permission(self, ai_billing_client):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=[])

        resp = await client.get("/api/v1/ai-billing/stats")
        assert resp.status_code == 403
        assert resp.json()["code"] == "A0301"

    async def test_anomalies_forbidden_without_permission(self, ai_billing_client):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=[])

        resp = await client.get("/api/v1/ai-billing/anomalies")
        assert resp.status_code == 403
        assert resp.json()["code"] == "A0301"

    async def test_anomalies_with_permission_reaches_service(self, ai_billing_client, monkeypatch):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=["ai:billing:stat"])

        received = {}

        async def fake_list(db, query):
            received["anomaly_type"] = query.anomaly_type
            received["page"] = query.page
            return PageResult(list=[], total=0)

        monkeypatch.setattr(billing_module.billing_anomaly_service, "list_anomalies", fake_list)

        resp = await client.get(
            "/api/v1/ai-billing/anomalies",
            params={"anomalyType": "single_high", "pageNum": 1, "pageSize": 10},
        )
        assert resp.status_code == 200
        assert received["anomaly_type"] == "single_high"

    async def test_costs_forbidden_without_permission(self, ai_billing_client):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=[])

        resp = await client.get("/api/v1/ai-billing/costs")
        assert resp.status_code == 403
        assert resp.json()["code"] == "A0301"

    async def test_cost_stats_forbidden_without_permission(self, ai_billing_client):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=[])

        resp = await client.get("/api/v1/ai-billing/cost-stats")
        assert resp.status_code == 403
        assert resp.json()["code"] == "A0301"

    async def test_reconcile_import_forbidden_without_permission(self, ai_billing_client):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=[])

        resp = await client.post(
            "/api/v1/ai-billing/reconcile/import",
            json={"content": "row", "startTime": "2026-08-01", "endTime": "2026-08-31"},
        )
        assert resp.status_code == 403
        assert resp.json()["code"] == "A0301"


class TestBillEndpointValidation:
    async def test_bill_invalid_month_rejected(self, ai_billing_client):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1)

        resp = await client.get("/api/v1/ai-billing/bills/invalid-month")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_bill_download_invalid_month_rejected(self, ai_billing_client):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1)

        resp = await client.get("/api/v1/ai-billing/bills/2026-13/download")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"


class TestBalancePerformanceSmoke:
    """性能烟测：余额查询（服务 mock 下）响应 < 1s"""

    async def test_balance_query_latency(self, ai_billing_client, monkeypatch):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1)
        monkeypatch.setattr(billing_module, "_build_balance", _fake_balance)

        start = time.perf_counter()
        resp = await client.get("/api/v1/ai-billing/balance")
        elapsed = time.perf_counter() - start
        assert resp.status_code == 200
        assert elapsed < 1.0, f"余额查询耗时 {elapsed:.3f}s 超出烟测阈值"


class TestStatsGroupByWhitelist:
    """GET /ai-billing/stats 的 groupBy：非法维度须落 A0400，不得 500 泄漏内部异常"""

    async def test_invalid_group_by_returns_param_error(self, ai_billing_client):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=["ai:billing:stat"])

        resp = await client.get("/api/v1/ai-billing/stats", params={"groupBy": "bill_type"})

        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_supported_group_by_reaches_service(self, ai_billing_client, monkeypatch):
        client, state = ai_billing_client
        state["user"] = _FakeUser(id=1, permissions=["ai:billing:stat"])
        captured = {}

        async def _fake_stats(db, query, user_id=None):
            captured["group_by"] = query.group_by
            return []

        monkeypatch.setattr(billing_module.billing_stat_service, "stats", _fake_stats)

        resp = await client.get("/api/v1/ai-billing/stats", params={"groupBy": "day"})

        assert resp.status_code == 200
        assert resp.json()["data"] == []
        assert captured["group_by"] == "day"
