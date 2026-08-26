import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.infrastructure.a2a.a2a_server import a2a_server
from app.main import app as fastapi_app

pytestmark = pytest.mark.api


class _FakeUser:
    def __init__(self, is_root=False, permissions=(), is_m2m=False):
        self.is_root = is_root
        self.permissions = list(permissions)
        self.is_m2m = is_m2m


@pytest.fixture
async def a2a_client():
    async def _override_db():
        return object()

    current_user = {"user": _FakeUser(is_m2m=True)}

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


async def test_global_a2a_paths_registered(app):
    schema = app.openapi()
    paths = schema["paths"]
    assert "/a2a" in paths and "post" in paths["/a2a"]
    assert "/.well-known/agent.json" in paths and "get" in paths["/.well-known/agent.json"]


async def test_global_a2a_resolves_agent_and_forwards(a2a_client, monkeypatch):
    client, state = a2a_client
    captured = {}

    async def fake_resolve(db, agent_id, agent_code):
        captured["resolved"] = (agent_id, agent_code)
        return 7

    async def fake_handle(db, redis, rpc, agent_id):
        captured["handled"] = (rpc.method, agent_id)
        from app.infrastructure.a2a.a2a_protocol import JsonRpcResponse

        return JsonRpcResponse(id=rpc.id, result={"ok": True})

    monkeypatch.setattr(a2a_server, "resolve_agent", fake_resolve)
    monkeypatch.setattr(a2a_server, "handle", fake_handle)
    resp = await client.post(
        "/a2a",
        json={"jsonrpc": "2.0", "id": 1, "method": "tasks/list", "params": {"agentId": 7}},
    )
    assert resp.status_code == 200
    assert resp.json()["result"]["ok"] is True
    assert captured["resolved"] == (7, None)
    assert captured["handled"] == ("tasks/list", 7)


async def test_global_a2a_resolves_by_code(a2a_client, monkeypatch):
    client, _ = a2a_client
    captured = {}

    async def fake_resolve(db, agent_id, agent_code):
        captured["resolved"] = (agent_id, agent_code)
        return 3

    async def fake_handle(db, redis, rpc, agent_id):
        from app.infrastructure.a2a.a2a_protocol import JsonRpcResponse

        return JsonRpcResponse(id=rpc.id, result={})

    monkeypatch.setattr(a2a_server, "resolve_agent", fake_resolve)
    monkeypatch.setattr(a2a_server, "handle", fake_handle)
    await client.post(
        "/a2a",
        json={"jsonrpc": "2.0", "id": 2, "method": "tasks/list", "params": {"agentCode": "dehaze"}},
    )
    assert captured["resolved"] == (None, "dehaze")


async def test_global_a2a_missing_selector_returns_404(a2a_client, monkeypatch):
    client, _ = a2a_client

    async def fake_resolve(db, agent_id, agent_code):
        raise ValueError("缺少 agent_id/agentCode 定位目标 Agent")

    monkeypatch.setattr(a2a_server, "resolve_agent", fake_resolve)
    resp = await client.post(
        "/a2a",
        json={"jsonrpc": "2.0", "id": 3, "method": "tasks/list", "params": {}},
    )
    assert resp.status_code == 404
    assert "agent_id/agentCode" in resp.json()["error"]["message"]


async def test_global_agent_card_returns_default(a2a_client, monkeypatch):
    client, _ = a2a_client
    card = {"name": "默认Agent", "version": "1", "url": "https://x/a2a", "capabilities": {}}

    async def fake_default(db):
        return 9

    async def fake_card(db, redis, agent_id, base_url):
        return card

    monkeypatch.setattr(a2a_server, "resolve_default_exposed_agent", fake_default)
    monkeypatch.setattr(a2a_server, "build_agent_card", fake_card)
    resp = await client.get("/.well-known/agent.json")
    assert resp.status_code == 200
    assert resp.json()["name"] == "默认Agent"
