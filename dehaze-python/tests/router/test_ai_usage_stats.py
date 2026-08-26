import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db

pytestmark = pytest.mark.api
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.service.ai_usage_stats_service import ai_usage_stats_service


class _FakeUser:
    def __init__(self, is_root=False, permissions=()):
        self.is_root = is_root
        self.permissions = list(permissions)


@pytest.fixture
async def ai_client():
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


async def test_usage_stats_path_registered(app):
    schema = app.openapi()
    assert "/api/v1/ai/usage/stats" in schema["paths"]
    assert "get" in schema["paths"]["/api/v1/ai/usage/stats"]


async def test_usage_stats_normal_user_forbidden(ai_client):
    client, state = ai_client
    state["user"] = _FakeUser(is_root=False, permissions=[])
    resp = await client.get("/api/v1/ai/usage/stats")
    assert resp.status_code == 403


async def test_usage_stats_admin_reaches_service(ai_client, monkeypatch):
    client, state = ai_client
    state["user"] = _FakeUser(is_root=True)

    async def fake_stats(db, redis, query):
        return {
            "provider_health": [],
            "model_usage": [],
            "degrade_fault": {"downgrade_frequency": [], "key_failover_count": 0},
        }

    monkeypatch.setattr(ai_usage_stats_service, "get_usage_stats", fake_stats)
    resp = await client.get("/api/v1/ai/usage/stats")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["providerHealth"] == []
    assert data["modelUsage"] == []
    assert data["degradeFault"]["keyFailoverCount"] == 0
