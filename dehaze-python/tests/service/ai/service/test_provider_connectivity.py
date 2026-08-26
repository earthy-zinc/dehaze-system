import re
from types import SimpleNamespace

import httpx
import pytest
import respx

from app.service.ai.service import provider_connectivity_service as conn_svc

# 连通性探测按协议拼接 URL：OpenAI 兼容 = {base}/models，Anthropic = {base}/v1/models。
OPENAI_URL_RE = re.compile(r"https://api\.openai\.com/v1/models")
ANTHROPIC_URL_RE = re.compile(r"https://api\.anthropic\.com/v1/models")


def _provider(**kwargs):
    fields = {
        "id": kwargs.get("id", 1),
        "api_base_url": kwargs.get("api_base_url", "https://api.openai.com/v1"),
        "protocol_type": kwargs.get("protocol_type", "openai_compat"),
        "auth_type": kwargs.get("auth_type", "bearer"),
        "default_headers": kwargs.get("default_headers"),
    }
    return SimpleNamespace(**fields)


async def _get_by_id(db, pid):
    return _provider_cache[pid]


async def _select_key(db, redis, provider_id):
    return "sk-test"


@pytest.fixture
def monkeypatch(monkeypatch):
    # 注册到 conftest autouse mock_redis 之外的仓储桩（非 httpx 依赖，保留显式桩）
    global _provider_cache
    _provider_cache = {}
    monkeypatch.setattr(conn_svc.ai_provider_repository, "get_by_id", _get_by_id)
    monkeypatch.setattr(conn_svc.provider_key_selector, "select_key", _select_key)
    return monkeypatch


async def test_openai_success(monkeypatch):
    _provider_cache[1] = _provider()
    with respx.mock(assert_all_mocked=True) as router:
        router.get(OPENAI_URL_RE).mock(return_value=httpx.Response(200))
        result = await conn_svc.test_connection(db=None, redis=None, provider_id=1)
    assert result["connected"] is True
    assert result["status_code"] == 200
    assert result["latency_ms"] is not None
    assert result["error"] is None


async def test_openai_unauthorized(monkeypatch):
    _provider_cache[1] = _provider()
    with respx.mock(assert_all_mocked=True) as router:
        router.get(OPENAI_URL_RE).mock(return_value=httpx.Response(401))
        result = await conn_svc.test_connection(db=None, redis=None, provider_id=1)
    assert result["connected"] is False
    assert result["status_code"] == 401
    assert result["error"] == "HTTP 401"


async def test_timeout(monkeypatch):
    _provider_cache[1] = _provider()
    with respx.mock(assert_all_mocked=True) as router:
        router.get(OPENAI_URL_RE).mock(side_effect=httpx.ConnectTimeout("timeout"))
        result = await conn_svc.test_connection(db=None, redis=None, provider_id=1)
    assert result["connected"] is False
    assert "超时" in result["error"]


async def test_connect_error(monkeypatch):
    _provider_cache[1] = _provider()
    with respx.mock(assert_all_mocked=True) as router:
        router.get(OPENAI_URL_RE).mock(side_effect=httpx.ConnectError("connect error"))
        result = await conn_svc.test_connection(db=None, redis=None, provider_id=1)
    assert result["connected"] is False
    assert result["error"] is not None
    assert "ConnectError" in result["error"]


async def test_anthropic_probe_headers(monkeypatch):
    _provider_cache[1] = _provider(
        protocol_type="anthropic",
        auth_type="x-api-key",
        api_base_url="https://api.anthropic.com",
        default_headers={"anthropic-version": "2023-06-01"},
    )
    captured = {}

    def _handler(request):
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        return httpx.Response(200)

    with respx.mock(assert_all_mocked=True) as router:
        router.get(ANTHROPIC_URL_RE).mock(side_effect=_handler)
        await conn_svc.test_connection(db=None, redis=None, provider_id=1)
    assert captured["url"] == "https://api.anthropic.com/v1/models"
    assert captured["headers"]["x-api-key"] == "sk-test"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
