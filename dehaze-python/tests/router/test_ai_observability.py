"""AI 可观测性查询路由测试：路径注册 / 权限校验 / 参数校验 / camelCase 序列化 / 导出"""
import pytest
from fastapi.responses import StreamingResponse
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.api

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.models.schema.ai_observability import CostsResult, SummaryResult, TraceDetailResult
from app.models.schema.common import PageResult
from app.service.ai_observability_service import ai_observability_service

AUDIT_PERMS = ["ai:conversation:audit"]
_ADMIN_PATHS = (
    "/api/v1/ai/observability/summary",
    "/api/v1/ai/observability/traces",
    "/api/v1/ai/observability/traces/export",
    "/api/v1/ai/observability/costs",
    "/api/v1/ai/observability/trends",
)


class _FakeUser:
    def __init__(self, id=1, is_root=False, permissions=()):
        self.id = id
        self.is_root = is_root
        self.permissions = list(permissions)


def _fake_trace_detail(trace_id: str = "tr-1") -> TraceDetailResult:
    return TraceDetailResult(
        trace_id=trace_id,
        conversation_id=1,
        status=1,
        duration_ms=10,
        llm_call_count=0,
        total_tokens=0,
        prompt_tokens=0,
        completion_tokens=0,
        cached_tokens=0,
        step_count=0,
    )


@pytest.fixture
async def obs_client():
    async def _override_db():
        return object()

    current_user = {"user": _FakeUser(permissions=AUDIT_PERMS)}

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


def test_observability_paths_registered(app):
    schema = app.openapi()
    for path in (
        *_ADMIN_PATHS,
        "/api/v1/ai/observability/traces/{trace_id}",
    ):
        assert path in schema["paths"], f"缺少路径 {path}"


@pytest.mark.parametrize("path", _ADMIN_PATHS)
async def test_admin_endpoints_forbidden_without_permission(path, obs_client):
    client, state = obs_client
    state["user"] = _FakeUser(id=1, permissions=[])
    resp = await client.get(path)
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_summary_wire_and_camel_case(obs_client, monkeypatch):
    client, _ = obs_client

    async def _fake_summary(db):
        return SummaryResult(
            total=10,
            success_count=6,
            failed_count=2,
            interrupted_count=1,
            timeout_count=1,
            quota_rejected=3,
            high_risk_calls=1,
        )

    monkeypatch.setattr(ai_observability_service, "summary", _fake_summary)
    resp = await client.get("/api/v1/ai/observability/summary")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["successCount"] == 6
    assert data["interruptedCount"] == 1
    assert data["quotaRejected"] == 3
    assert data["highRiskCalls"] == 1


async def test_list_traces_query_wire(obs_client, monkeypatch):
    client, _ = obs_client
    captured = {}

    async def _fake_list(db, query):
        captured["query"] = query
        return PageResult(list=[], total=0)

    monkeypatch.setattr(ai_observability_service, "list_traces", _fake_list)
    resp = await client.get(
        "/api/v1/ai/observability/traces",
        params={
            "conversationId": 5,
            "userId": 7,
            "status": 2,
            "agentCode": "a1",
            "model": "m1",
            "errorType": "quota",
            "keyword": "tr-1",
            "capability": "memory",
            "pageNum": 2,
            "pageSize": 5,
        },
    )
    assert resp.status_code == 200
    q = captured["query"]
    assert (q.conversationId, q.userId, q.status) == (5, 7, 2)
    assert (q.agentCode, q.model) == ("a1", "m1")
    assert (q.errorType, q.keyword, q.capability) == ("quota", "tr-1", "memory")
    assert (q.pageNum, q.pageSize) == (2, 5)


async def test_list_traces_invalid_status(obs_client):
    client, _ = obs_client
    resp = await client.get("/api/v1/ai/observability/traces", params={"status": 5})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_detail_admin_flag_passed(obs_client, monkeypatch):
    client, state = obs_client
    state["user"] = _FakeUser(id=9, permissions=AUDIT_PERMS)
    captured = {}

    async def _fake_get_trace(db, trace_id, user_id, *, admin):
        captured.update(trace_id=trace_id, user_id=user_id, admin=admin)
        return _fake_trace_detail()

    monkeypatch.setattr(ai_observability_service, "get_trace", _fake_get_trace)
    resp = await client.get("/api/v1/ai/observability/traces/tr-1")
    assert resp.status_code == 200
    assert captured == {"trace_id": "tr-1", "user_id": 9, "admin": True}
    data = resp.json()["data"]
    assert data["traceId"] == "tr-1"
    assert data["thoughts"] == []
    assert data["messages"] == []


async def test_detail_normal_user_allowed_not_admin(obs_client, monkeypatch):
    client, state = obs_client
    state["user"] = _FakeUser(id=7, permissions=[])

    async def _fake_get_trace(db, trace_id, user_id, *, admin):
        assert admin is False
        return _fake_trace_detail()

    monkeypatch.setattr(ai_observability_service, "get_trace", _fake_get_trace)
    resp = await client.get("/api/v1/ai/observability/traces/tr-1")
    assert resp.status_code == 200


async def test_export_streams_csv(obs_client, monkeypatch):
    client, state = obs_client
    state["user"] = _FakeUser(id=1, permissions=AUDIT_PERMS)

    async def _fake_export(db, query):
        return StreamingResponse(
            iter(["trace_id,status\ntr-1,1\n"]),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="ai_traces.csv"'},
        )

    monkeypatch.setattr(ai_observability_service, "export_traces", _fake_export)
    resp = await client.get("/api/v1/ai/observability/traces/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert "ai_traces.csv" in resp.headers["content-disposition"]
    assert "tr-1" in resp.text


async def test_costs_query_wire(obs_client, monkeypatch):
    client, _ = obs_client
    captured = {}

    async def _fake_costs(db, query):
        captured["query"] = query
        return CostsResult(items=[], total=0, trend=[])

    monkeypatch.setattr(ai_observability_service, "costs", _fake_costs)
    resp = await client.get(
        "/api/v1/ai/observability/costs",
        params={"dimension": "user", "pageNum": 2, "pageSize": 5},
    )
    assert resp.status_code == 200
    q = captured["query"]
    assert q.dimension == "user"
    assert (q.pageNum, q.pageSize) == (2, 5)


async def test_costs_invalid_dimension(obs_client):
    client, _ = obs_client
    resp = await client.get("/api/v1/ai/observability/costs", params={"dimension": "day"})
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_trends_query_wire(obs_client, monkeypatch):
    client, _ = obs_client
    captured = {}

    async def _fake_trends(db, query):
        captured["query"] = query
        return []

    monkeypatch.setattr(ai_observability_service, "trends", _fake_trends)
    resp = await client.get("/api/v1/ai/observability/trends", params={"dimension": "agent"})
    assert resp.status_code == 200
    assert captured["query"].dimension == "agent"
