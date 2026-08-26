"""外部 MCP Server 管理路由测试（F-M08-006 §2.6.13）。

构造注入 monkeypatch service，只断言业务结果；权限：管理操作 `ai:mcp:manage`，
普通用户越权返回 403（A0301）。
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db

pytestmark = pytest.mark.api
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.models.schema.ai_mcp import McpHealthResult, McpServerResult
from app.models.schema.common import PageResult
from app.router import ai_mcp


class _FakeUser:
    def __init__(self, is_root=False, permissions=()):
        self.is_root = is_root
        self.permissions = list(permissions)


def _server(**overrides) -> McpServerResult:
    base = dict(
        id=1,
        name="测试Server",
        description="描述",
        protocolType="streamable-http",
        endpoint="https://example.com/mcp",
        authType="api_key",
        status=1,
        health=None,
        toolCount=0,
    )
    base.update(overrides)
    return McpServerResult(**base)


@pytest.fixture
async def mcp_client():
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


async def test_mcp_paths_registered(app):
    schema = app.openapi()
    paths = schema["paths"]
    assert "/api/v1/ai/mcp/servers" in paths
    assert "get" in paths["/api/v1/ai/mcp/servers"]
    assert "post" in paths["/api/v1/ai/mcp/servers"]
    assert "put" in paths["/api/v1/ai/mcp/servers/{server_id}"]
    assert "delete" in paths["/api/v1/ai/mcp/servers/{server_id}"]
    assert "patch" in paths["/api/v1/ai/mcp/servers/{server_id}/status"]
    assert "get" in paths["/api/v1/ai/mcp/servers/{server_id}/health"]
    assert "get" in paths["/api/v1/ai/mcp/servers/{server_id}/tools"]
    assert "get" in paths["/api/v1/ai/mcp/servers/{server_id}/namespaces"]
    assert "put" in paths["/api/v1/ai/mcp/servers/{server_id}/credentials"]
    assert "get" in paths["/api/v1/ai/mcp/market"]
    assert "post" in paths["/api/v1/ai/mcp/market/{preset_id}/install"]
    assert "get" in paths["/api/v1/ai/mcp/calls"]


async def test_list_servers_returns_page(mcp_client, monkeypatch):
    client, state = mcp_client
    state["user"] = _FakeUser(is_root=True)
    captured = {}

    async def fake_list(db, page, size, keyword, status):
        captured.update(page=page, size=size, keyword=keyword, status=status)
        return [_server()], 1

    monkeypatch.setattr(ai_mcp.ai_mcp_server_service, "list_servers", fake_list)
    resp = await client.get("/api/v1/ai/mcp/servers?pageNum=1&pageSize=10&keyword=测&status=1")
    assert resp.status_code == 200
    assert resp.json()["data"]["total"] == 1
    assert captured == {"page": 1, "size": 10, "keyword": "测", "status": 1}


async def test_create_server_admin_success(mcp_client, monkeypatch):
    client, state = mcp_client
    state["user"] = _FakeUser(is_root=True)
    captured = {}

    async def fake_create(db, form):
        captured["name"] = form.name
        return _server(name=form.name)

    monkeypatch.setattr(ai_mcp.ai_mcp_server_service, "create_server", fake_create)
    resp = await client.post(
        "/api/v1/ai/mcp/servers",
        json={"name": "新Server", "protocolType": "streamable-http", "endpoint": "https://x.com/mcp"},
    )
    assert resp.status_code == 200
    assert captured["name"] == "新Server"


async def test_create_server_normal_user_forbidden(mcp_client, monkeypatch):
    client, state = mcp_client
    state["user"] = _FakeUser(is_root=False, permissions=[])
    resp = await client.post("/api/v1/ai/mcp/servers", json={"name": "x", "protocolType": "sse"})
    assert resp.status_code == 403


async def test_get_server_detail(mcp_client, monkeypatch):
    client, state = mcp_client
    state["user"] = _FakeUser(is_root=True)
    captured = {}

    async def fake_get(db, server_id):
        captured["server_id"] = server_id
        return _server(id=server_id)

    monkeypatch.setattr(ai_mcp.ai_mcp_server_service, "get_server", fake_get)
    resp = await client.get("/api/v1/ai/mcp/servers/7")
    assert resp.status_code == 200
    assert resp.json()["data"]["id"] == 7
    assert captured["server_id"] == 7


async def test_update_server(mcp_client, monkeypatch):
    client, state = mcp_client
    state["user"] = _FakeUser(is_root=True)
    captured = {}

    async def fake_update(db, server_id, form):
        captured["server_id"] = server_id
        captured["description"] = form.description
        return _server(id=server_id, description=form.description)

    monkeypatch.setattr(ai_mcp.ai_mcp_server_service, "update_server", fake_update)
    resp = await client.put("/api/v1/ai/mcp/servers/7", json={"description": "新描述"})
    assert resp.status_code == 200
    assert captured["server_id"] == 7
    assert captured["description"] == "新描述"


async def test_delete_server(mcp_client, monkeypatch):
    client, state = mcp_client
    state["user"] = _FakeUser(is_root=True)
    captured = {}

    async def fake_delete(db, server_id):
        captured["server_id"] = server_id

    monkeypatch.setattr(ai_mcp.ai_mcp_server_service, "delete_server", fake_delete)
    resp = await client.delete("/api/v1/ai/mcp/servers/7")
    assert resp.status_code == 200
    assert captured["server_id"] == 7


async def test_switch_server_status(mcp_client, monkeypatch):
    client, state = mcp_client
    state["user"] = _FakeUser(is_root=True)
    captured = {}

    async def fake_switch(db, server_id, status):
        captured["server_id"] = server_id
        captured["status"] = status
        return _server(id=server_id, status=status)

    monkeypatch.setattr(ai_mcp.ai_mcp_server_service, "switch_server_status", fake_switch)
    resp = await client.patch("/api/v1/ai/mcp/servers/7/status", json={"status": 0})
    assert resp.status_code == 200
    assert resp.json()["data"]["status"] == 0
    assert captured == {"server_id": 7, "status": 0}


async def test_probe_health(mcp_client, monkeypatch):
    client, state = mcp_client
    state["user"] = _FakeUser(is_root=True)
    captured = {}

    async def fake_probe(db, server_id):
        captured["server_id"] = server_id
        return McpHealthResult(status="online", latency_ms=10)

    monkeypatch.setattr(ai_mcp.ai_mcp_server_service, "probe_health", fake_probe)
    resp = await client.get("/api/v1/ai/mcp/servers/7/health")
    assert resp.status_code == 200
    assert resp.json()["data"]["status"] == "online"
    assert captured["server_id"] == 7


async def test_get_tools(mcp_client, monkeypatch):
    client, state = mcp_client
    state["user"] = _FakeUser(is_root=True)
    captured = {}

    async def fake_tools(db, server_id):
        captured["server_id"] = server_id
        return []

    monkeypatch.setattr(ai_mcp.mcp_manage_service, "get_tools", fake_tools)
    resp = await client.get("/api/v1/ai/mcp/servers/7/tools")
    assert resp.status_code == 200
    assert resp.json()["data"] == []
    assert captured["server_id"] == 7


async def test_list_namespaces(mcp_client, monkeypatch):
    client, state = mcp_client
    state["user"] = _FakeUser(is_root=True)

    async def fake_namespaces(db, server_id):
        return []

    monkeypatch.setattr(ai_mcp.ai_mcp_server_service, "list_namespaces", fake_namespaces)
    resp = await client.get("/api/v1/ai/mcp/servers/7/namespaces")
    assert resp.status_code == 200
    assert resp.json()["data"] == []


async def test_update_namespaces(mcp_client, monkeypatch):
    client, state = mcp_client
    state["user"] = _FakeUser(is_root=True)
    captured = {}

    async def fake_update(db, server_id, namespaces):
        captured["server_id"] = server_id
        captured["names"] = [n.name for n in namespaces]
        return [{"name": n.name, "toolNames": n.toolNames} for n in namespaces]

    monkeypatch.setattr(ai_mcp.ai_mcp_server_service, "update_namespaces", fake_update)
    resp = await client.put(
        "/api/v1/ai/mcp/servers/7/namespaces",
        json=[{"name": "ns_a", "toolNames": ["tool_a"]}],
    )
    assert resp.status_code == 200
    assert captured == {"server_id": 7, "names": ["ns_a"]}
    assert resp.json()["data"] == [{"name": "ns_a", "toolNames": ["tool_a"]}]


async def test_update_namespaces_normal_user_forbidden(mcp_client, monkeypatch):
    client, state = mcp_client
    state["user"] = _FakeUser(is_root=False, permissions=[])
    resp = await client.put(
        "/api/v1/ai/mcp/servers/7/namespaces", json=[{"name": "ns_a", "toolNames": []}]
    )
    assert resp.status_code == 403


async def test_update_credentials(mcp_client, monkeypatch):
    client, state = mcp_client
    state["user"] = _FakeUser(is_root=True)
    captured = {}

    async def fake_cred(db, server_id, form):
        captured["server_id"] = server_id
        captured["api_key"] = form.api_key

    monkeypatch.setattr(ai_mcp.ai_mcp_server_service, "update_credentials", fake_cred)
    resp = await client.put(
        "/api/v1/ai/mcp/servers/7/credentials", json={"apiKey": "test_secret_key"}
    )
    assert resp.status_code == 200
    assert captured == {"server_id": 7, "api_key": "test_secret_key"}


async def test_get_market(mcp_client, monkeypatch):
    client, state = mcp_client
    state["user"] = _FakeUser(is_root=True)

    async def fake_market(db):
        return []

    monkeypatch.setattr(ai_mcp.mcp_manage_service, "get_market", fake_market)
    resp = await client.get("/api/v1/ai/mcp/market")
    assert resp.status_code == 200
    assert resp.json()["data"] == []


async def test_install_preset(mcp_client, monkeypatch):
    client, state = mcp_client
    state["user"] = _FakeUser(is_root=True)
    captured = {}

    async def fake_install(db, redis, preset_id):
        captured["preset_id"] = preset_id
        return _server(name="GitHub")

    monkeypatch.setattr(ai_mcp.mcp_manage_service, "install_preset", fake_install)
    resp = await client.post("/api/v1/ai/mcp/market/github/install")
    assert resp.status_code == 200
    assert resp.json()["data"]["name"] == "GitHub"
    assert captured["preset_id"] == "github"


async def test_list_calls(mcp_client, monkeypatch):
    client, state = mcp_client
    state["user"] = _FakeUser(is_root=True)
    captured = {}

    async def fake_calls(db, query):
        captured["page"] = query.pageNum
        return PageResult(list=[], total=0)

    monkeypatch.setattr(ai_mcp.mcp_manage_service, "list_calls", fake_calls)
    resp = await client.get("/api/v1/ai/mcp/calls?pageNum=1&pageSize=10")
    assert resp.status_code == 200
    assert resp.json()["data"]["total"] == 0
    assert captured["page"] == 1


async def test_list_calls_normal_user_forbidden(mcp_client, monkeypatch):
    client, state = mcp_client
    state["user"] = _FakeUser(is_root=False, permissions=[])
    resp = await client.get("/api/v1/ai/mcp/calls")
    assert resp.status_code == 403
