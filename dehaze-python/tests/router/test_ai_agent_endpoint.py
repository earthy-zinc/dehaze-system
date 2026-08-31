"""外部 A2A 端点管理路由测试（/api/v1/ai/a2a/endpoints）

覆盖重点：路由注册、ai:agent:manage 权限拦截（A0301）、参数校验（A0400）、
端点 CRUD 与 Agent Card 手动刷新、camelCase 序列化。
"""
from datetime import datetime
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.api

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.service.ai_agent_endpoint_service import ai_agent_endpoint_service

MANAGE_PERM = "ai:agent:manage"
_ENDPOINTS = "/api/v1/ai/a2a/endpoints"


class _FakeUser:
    def __init__(self, id=1, is_root=False, permissions=(MANAGE_PERM,)):
        self.id = id
        self.is_root = is_root
        self.permissions = list(permissions)


def _endpoint(**overrides):
    base = {
        "id": 4,
        "name": "图像算法Agent",
        "agent_card_url": "https://remote.example.com/.well-known/agent.json",
        "base_url": "https://remote.example.com/a2a",
        "auth_type": "http",
        "agent_card": {"name": "图像算法Agent", "version": "1.0"},
        "status": 1,
        "create_time": datetime(2026, 8, 29, 10, 0, 0),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
async def endpoint_client():
    async def _override_db():
        return object()

    current_user = {"user": _FakeUser(id=8)}

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


def test_endpoint_paths_registered(app):
    schema = app.openapi()
    for path in (_ENDPOINTS, f"{_ENDPOINTS}/{{endpoint_id}}",
                 f"{_ENDPOINTS}/{{endpoint_id}}/refresh-card"):
        assert path in schema["paths"], f"缺少路径 {path}"


_PERMISSION_CASES = [
    ("post", _ENDPOINTS, {"name": "n", "base_url": "https://a.example.com/a2a"}),
    ("get", _ENDPOINTS, None),
    ("patch", f"{_ENDPOINTS}/4", {"name": "n"}),
    ("delete", f"{_ENDPOINTS}/4", None),
    ("post", f"{_ENDPOINTS}/4/refresh-card", None),
]


@pytest.mark.parametrize(("method", "path", "body"), _PERMISSION_CASES)
async def test_endpoints_forbidden_without_permission(method, path, body, endpoint_client):
    client, state = endpoint_client
    state["user"] = _FakeUser(id=8, permissions=[])
    resp = await client.request(method, path, json=body)
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


class TestCreate:
    async def test_create_endpoint(self, endpoint_client, monkeypatch):
        client, _ = endpoint_client
        captured: dict = {}

        async def _fake_create(db, form):
            captured.update(name=form.name, base_url=form.base_url, auth_type=form.auth_type)
            return _endpoint()

        monkeypatch.setattr(ai_agent_endpoint_service, "create_endpoint", _fake_create)
        resp = await client.post(
            _ENDPOINTS,
            json={
                "name": "图像算法Agent",
                "base_url": "https://remote.example.com/a2a",
                "agent_card_url": "https://remote.example.com/.well-known/agent.json",
                "auth_type": "apiKey",
                "credential": "cipher",
            },
        )
        assert resp.status_code == 200
        assert captured == {
            "name": "图像算法Agent",
            "base_url": "https://remote.example.com/a2a",
            "auth_type": "apiKey",
        }
        data = resp.json()["data"]
        assert data["agentCardUrl"].endswith("/.well-known/agent.json")
        assert data["agentCard"]["version"] == "1.0"

    async def test_create_requires_base_url(self, endpoint_client):
        client, _ = endpoint_client
        resp = await client.post(_ENDPOINTS, json={"name": "图像算法Agent"})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_create_invalid_auth_type_rejected(self, endpoint_client):
        client, _ = endpoint_client
        resp = await client.post(
            _ENDPOINTS,
            json={"name": "n", "base_url": "https://a.example.com/a2a", "auth_type": "basic"},
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_create_status_out_of_range_rejected(self, endpoint_client):
        client, _ = endpoint_client
        resp = await client.post(
            _ENDPOINTS,
            json={"name": "n", "base_url": "https://a.example.com/a2a", "status": 2},
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_create_unsafe_url_maps_a0400(self, endpoint_client, monkeypatch):
        client, _ = endpoint_client

        async def _fake_create(db, form):
            raise BusinessException(
                ResultCode.PARAM_ERROR, "base_url/agent_card_url 仅支持 https 且禁止内网地址"
            )

        monkeypatch.setattr(ai_agent_endpoint_service, "create_endpoint", _fake_create)
        resp = await client.post(
            _ENDPOINTS, json={"name": "n", "base_url": "http://10.0.0.1/a2a"}
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"


class TestUpdateDelete:
    async def test_update_endpoint(self, endpoint_client, monkeypatch):
        client, _ = endpoint_client
        captured: dict = {}

        async def _fake_update(db, endpoint_id, form):
            captured.update(endpoint_id=endpoint_id, name=form.name, status=form.status)
            return _endpoint(id=endpoint_id, name=form.name, status=form.status)

        monkeypatch.setattr(ai_agent_endpoint_service, "update_endpoint", _fake_update)
        resp = await client.patch(f"{_ENDPOINTS}/4", json={"name": "改后名称", "status": 0})
        assert resp.status_code == 200
        assert captured == {"endpoint_id": 4, "name": "改后名称", "status": 0}
        assert resp.json()["data"]["status"] == 0

    async def test_update_missing_endpoint_maps_a0401(self, endpoint_client, monkeypatch):
        client, _ = endpoint_client

        async def _fake_update(db, endpoint_id, form):
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "端点不存在")

        monkeypatch.setattr(ai_agent_endpoint_service, "update_endpoint", _fake_update)
        resp = await client.patch(f"{_ENDPOINTS}/404", json={"name": "x"})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0401"

    async def test_delete_endpoint(self, endpoint_client, monkeypatch):
        client, _ = endpoint_client
        captured: dict = {}

        async def _fake_delete(db, endpoint_id):
            captured["endpoint_id"] = endpoint_id

        monkeypatch.setattr(ai_agent_endpoint_service, "delete_endpoint", _fake_delete)
        resp = await client.delete(f"{_ENDPOINTS}/4")
        assert resp.status_code == 200
        assert captured == {"endpoint_id": 4}


class TestList:
    async def test_list_forwards_filters(self, endpoint_client, monkeypatch):
        client, _ = endpoint_client
        captured: dict = {}

        async def _fake_list(db, page, size, keyword=None, status=None):
            captured.update(page=page, size=size, keyword=keyword, status=status)
            return [_endpoint()], 1

        monkeypatch.setattr(ai_agent_endpoint_service, "list_endpoints", _fake_list)
        resp = await client.get(
            _ENDPOINTS, params={"keyword": "图像", "status": 1, "pageNum": 2, "pageSize": 5}
        )
        assert resp.status_code == 200
        assert captured == {"page": 2, "size": 5, "keyword": "图像", "status": 1}
        data = resp.json()["data"]
        assert data["total"] == 1
        assert data["list"][0]["baseUrl"] == "https://remote.example.com/a2a"

    async def test_list_status_out_of_range_rejected(self, endpoint_client):
        client, _ = endpoint_client
        resp = await client.get(_ENDPOINTS, params={"status": 2})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_list_page_size_too_large_rejected(self, endpoint_client):
        client, _ = endpoint_client
        resp = await client.get(_ENDPOINTS, params={"pageSize": 101})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"


class TestRefreshCard:
    async def test_refresh_card(self, endpoint_client, monkeypatch):
        client, _ = endpoint_client
        captured: dict = {}

        async def _fake_refresh(db, endpoint_id):
            captured["endpoint_id"] = endpoint_id
            return {"name": "图像算法Agent", "version": "2.0", "status": "ok"}

        monkeypatch.setattr(ai_agent_endpoint_service, "refresh_agent_card", _fake_refresh)
        resp = await client.post(f"{_ENDPOINTS}/4/refresh-card")
        assert resp.status_code == 200
        assert captured == {"endpoint_id": 4}
        assert resp.json()["data"]["version"] == "2.0"

    async def test_refresh_missing_endpoint_maps_a0401(self, endpoint_client, monkeypatch):
        client, _ = endpoint_client

        async def _fake_refresh(db, endpoint_id):
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "端点不存在")

        monkeypatch.setattr(ai_agent_endpoint_service, "refresh_agent_card", _fake_refresh)
        resp = await client.post(f"{_ENDPOINTS}/404/refresh-card")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0401"

    async def test_refresh_without_card_url_maps_a0400(self, endpoint_client, monkeypatch):
        client, _ = endpoint_client

        async def _fake_refresh(db, endpoint_id):
            raise BusinessException(ResultCode.PARAM_ERROR, "端点未配置 agent_card_url")

        monkeypatch.setattr(ai_agent_endpoint_service, "refresh_agent_card", _fake_refresh)
        resp = await client.post(f"{_ENDPOINTS}/4/refresh-card")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"
