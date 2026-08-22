from types import SimpleNamespace

import httpx

from app.service.ai import provider_connectivity_service as conn_svc


class _FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _FakeAsyncClient:
    def __init__(self, behavior):
        self.behavior = behavior

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, headers=None):
        b = self.behavior
        if b.get("timeout"):
            raise httpx.TimeoutException("timeout")
        if b.get("error"):
            raise httpx.ConnectError("connect error")
        return _FakeResponse(b["status_code"])


async def _run(monkeypatch, behavior, provider, key="sk-test"):
    async def _get_by_id(db, pid):
        return provider

    async def _select_key(db, redis, provider_id):
        return key

    monkeypatch.setattr(conn_svc.ai_provider_repository, "get_by_id", _get_by_id)
    monkeypatch.setattr(conn_svc.ai_provider_key_service, "select_key", _select_key)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(behavior))
    return await conn_svc.test_connection(db=None, redis=None, provider_id=provider.id)


def _provider(**kwargs):
    fields = {
        "id": kwargs.get("id", 1),
        "api_base_url": kwargs.get("api_base_url", "https://api.openai.com/v1"),
        "protocol_type": kwargs.get("protocol_type", "openai_compat"),
        "auth_type": kwargs.get("auth_type", "bearer"),
        "default_headers": kwargs.get("default_headers"),
    }
    return SimpleNamespace(**fields)


async def test_openai_success(monkeypatch):
    result = await _run(monkeypatch, {"status_code": 200}, _provider())
    assert result["connected"] is True
    assert result["status_code"] == 200
    assert result["latency_ms"] is not None
    assert result["error"] is None


async def test_openai_unauthorized(monkeypatch):
    result = await _run(monkeypatch, {"status_code": 401}, _provider())
    assert result["connected"] is False
    assert result["status_code"] == 401
    assert result["error"] == "HTTP 401"


async def test_timeout(monkeypatch):
    result = await _run(monkeypatch, {"timeout": True}, _provider())
    assert result["connected"] is False
    assert "超时" in result["error"]


async def test_connect_error(monkeypatch):
    result = await _run(monkeypatch, {"error": True}, _provider())
    assert result["connected"] is False
    assert result["error"] is not None
    assert "ConnectError" in result["error"]


async def test_anthropic_probe_headers(monkeypatch):
    captured = {}

    class _Client(_FakeAsyncClient):
        def __init__(self):
            self.behavior = {"status_code": 200}

        async def get(self, url, headers=None):
            captured["url"] = url
            captured["headers"] = headers
            return _FakeResponse(200)

    async def _get_by_id(db, pid):
        return _provider(
            protocol_type="anthropic",
            auth_type="x-api-key",
            api_base_url="https://api.anthropic.com",
            default_headers={"anthropic-version": "2023-06-01"},
        )

    async def _select_key(db, redis, provider_id):
        return "sk-ant-test"

    monkeypatch.setattr(conn_svc.ai_provider_repository, "get_by_id", _get_by_id)
    monkeypatch.setattr(conn_svc.ai_provider_key_service, "select_key", _select_key)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _Client())

    await conn_svc.test_connection(db=None, redis=None, provider_id=1)
    assert captured["url"] == "https://api.anthropic.com/v1/models"
    assert captured["headers"]["x-api-key"] == "sk-ant-test"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
