"""AI 计费管理路由测试：userId 下钻查询与权限校验"""
from decimal import Decimal

import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.api

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.models.schema.ai_billing import BalanceResult
from app.models.schema.common import PageResult
from app.router import ai_billing as billing_module
from app.service.billing.billing_record_service import billing_record_service
from app.service.billing.refund_service import refund_service


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
