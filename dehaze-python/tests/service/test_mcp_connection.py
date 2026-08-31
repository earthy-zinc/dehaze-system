"""MCP 连接公共工具测试：凭据鉴权头构造 + 端点安全校验（SSRF 一致）"""

from app.infrastructure.crypto.aes_cipher import encrypt
from app.models.entity.sys_ai_mcp_server import SysAiMcpServer
from app.service.ai_mcp.mcp_connection import (
    apply_ssrf_guard,
    build_mcp_auth_headers,
    check_endpoint_safe,
)
from app.service.ai_mcp.mcp_presets import PRESET_ENDPOINTS


def _server(
    credentials=None,
    protocol_type="streamable-http",
    endpoint="https://api.example.com/mcp",
):
    return SysAiMcpServer(
        name="srv",
        description="",
        protocol_type=protocol_type,
        endpoint=endpoint,
        credentials=credentials,
        status=1,
        tool_count=0,
    )


class TestBuildAuthHeaders:
    def test_api_key_encrypted_to_bearer(self):
        server = _server(credentials={"api_key": encrypt("sk_123")})
        assert build_mcp_auth_headers(server) == {"Authorization": "Bearer sk_123"}

    def test_oauth2_access_token_from_extra(self):
        server = _server(credentials={"extra": {"access_token": encrypt("tok_9")}})
        assert build_mcp_auth_headers(server) == {"Authorization": "Bearer tok_9"}

    def test_no_credentials_returns_empty(self):
        assert build_mcp_auth_headers(_server(credentials=None)) == {}

    def test_bad_cipher_falls_back_empty(self):
        server = _server(credentials={"api_key": "not-valid-cipher!"})
        assert build_mcp_auth_headers(server) == {}


class TestCheckEndpointSafe:
    async def test_preset_endpoint_allowed(self):
        endpoint = next(iter(PRESET_ENDPOINTS))
        assert await check_endpoint_safe(endpoint) is True

    async def test_public_https_passes_ssrf(self, monkeypatch):
        async def _safe(url):
            return url.startswith("https://")

        monkeypatch.setattr("app.service.ai_mcp.mcp_connection.is_safe_url", _safe)
        assert await check_endpoint_safe("https://mcp.example.com") is True

    async def test_internal_rejected(self, monkeypatch):
        async def _safe(url):
            return False

        monkeypatch.setattr("app.service.ai_mcp.mcp_connection.is_safe_url", _safe)
        assert await check_endpoint_safe("http://127.0.0.1:9999/mcp") is False
        assert await check_endpoint_safe("http://10.0.0.5/mcp") is False


class TestApplySsrfGuard:
    async def test_unsafe_endpoint_rejected(self, monkeypatch):
        async def _safe(url):
            return False

        monkeypatch.setattr("app.service.ai_mcp.mcp_connection.is_safe_url", _safe)
        ok, reason = await apply_ssrf_guard(
            _server(endpoint="http://192.168.1.1/mcp")
        )
        assert not ok
        assert "不安全" in reason

    async def test_stdio_local_process_bypasses(self):
        ok, reason = await apply_ssrf_guard(_server(protocol_type="stdio", endpoint=None))
        assert ok is True
        assert reason == ""

    async def test_missing_endpoint_rejected(self):
        ok, _ = await apply_ssrf_guard(_server(endpoint=None))
        assert ok is False

    async def test_preset_endpoint_allowed(self):
        ok, _ = await apply_ssrf_guard(_server(endpoint=next(iter(PRESET_ENDPOINTS))))
        assert ok is True
