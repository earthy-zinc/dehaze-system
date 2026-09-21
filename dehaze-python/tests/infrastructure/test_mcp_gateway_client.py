"""McpGatewayClient 单元测试（mock 底层 MCP 传输与会话类，不依赖真实网关）。

回归重点：MCP 协议要求 ClientSession 必须先 initialize 再 tools/call，
缺失时网关以 400 拒绝、调用挂起后被 cancel scope 取消（曾导致全链路静默失效）。
mock 只替换模块级 streamablehttp_client / ClientSession，真实 _McpSessionContext
的会话装配逻辑（含 initialize 前置）被执行。
"""

import asyncio
import json

from app.infrastructure.clients import mcp_gateway_client as mod


class _FakeClientSession:
    def __init__(self, read, write, fail_enter=False, list_delay=0.0):
        self.fail_enter = fail_enter
        self.list_delay = list_delay
        self.initialized = False
        self.calls = []

    async def __aenter__(self):
        if self.fail_enter:
            raise ConnectionError("refused")
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self):
        self.initialized = True

    async def call_tool(self, name, args):
        assert self.initialized, "call_tool 前必须完成 initialize 能力协商"
        self.calls.append((name, args))
        if name == "boom":
            raise ConnectionError("gateway down")

        class _Block:
            text = f"text-of-{name}"

        class _Result:
            content = [_Block()]

        return _Result()

    async def list_tools(self):
        if self.list_delay:
            await asyncio.sleep(self.list_delay)
        assert self.initialized, "list_tools 前必须完成 initialize 能力协商"

        class _Tool:
            name = "lookup_tool"
            description = "搜索"
            inputSchema = {"type": "object"}

        class _Result:
            tools = [_Tool()]

        return _Result()


def _install(monkeypatch, session_kwargs, gateway_key="test-key"):
    created = []
    http_headers = []

    monkeypatch.setattr(mod.settings, "MCP_GATEWAY_KEY", gateway_key)

    def fake_http_client(url, headers=None):
        http_headers.append(headers)

        class _HttpCtx:
            async def __aenter__(self):
                return ("read", "write", None)

            async def __aexit__(self, *exc):
                return False

        return _HttpCtx()

    def fake_client_session(read, write):
        session = _FakeClientSession(read, write, **session_kwargs)
        created.append(session)
        return session

    monkeypatch.setattr(mod, "streamablehttp_client", fake_http_client)
    monkeypatch.setattr(mod, "ClientSession", fake_client_session)
    return created, http_headers


def test_initialize_before_call(monkeypatch):
    created, _ = _install(monkeypatch, {})
    out = asyncio.run(mod.mcp_gateway_client.lookup_tool("预测"))
    assert created[0].initialized
    assert out == "text-of-lookup_tool"


def test_gateway_key_header_sent(monkeypatch):
    """每个会话必须经 X-MCP-Key 请求头携带网关共享密钥。"""
    _, http_headers = _install(monkeypatch, {})
    asyncio.run(mod.mcp_gateway_client.lookup_tool("预测"))
    assert http_headers[0] == {"X-MCP-Key": "test-key"}


def test_missing_gateway_key_rejected(monkeypatch):
    """密钥未配置直接拒绝连接（fail fast），不发起任何 MCP 会话。"""
    created, http_headers = _install(monkeypatch, {}, gateway_key="")
    out = asyncio.run(mod.mcp_gateway_client.lookup_tool("预测"))
    assert out == ""
    assert created == []
    assert http_headers == []


def test_lookup_down_returns_empty(monkeypatch):
    _install(monkeypatch, {"fail_enter": True})
    assert asyncio.run(mod.mcp_gateway_client.lookup_tool("x")) == ""


def test_execute_tool_serializes_arguments(monkeypatch):
    created, _ = _install(monkeypatch, {})
    args = {"pageNum": 1}
    asyncio.run(mod.mcp_gateway_client.execute_tool("get_x", args))
    name, sent = created[0].calls[0]
    assert name == "execute_tool"
    assert json.loads(sent["arguments"]) == args


def test_execute_tool_down_returns_error_text(monkeypatch):
    _install(monkeypatch, {"fail_enter": True})
    out = asyncio.run(mod.mcp_gateway_client.execute_tool("get_x", {}))
    assert "工具执行失败" in out


def test_list_tools_returns_summaries(monkeypatch):
    _install(monkeypatch, {})
    tools = asyncio.run(mod.mcp_gateway_client.list_tools())
    assert tools == [
        {"name": "lookup_tool", "description": "搜索", "input_schema": {"type": "object"}}
    ]


def test_list_tools_down_returns_empty(monkeypatch):
    _install(monkeypatch, {"fail_enter": True})
    assert asyncio.run(mod.mcp_gateway_client.list_tools()) == []


def test_list_tools_timeout_returns_empty(monkeypatch):
    _install(monkeypatch, {"list_delay": 5.0})
    monkeypatch.setattr(mod, "_MCP_LIST_TIMEOUT", 0.1)
    assert asyncio.run(mod.mcp_gateway_client.list_tools()) == []
