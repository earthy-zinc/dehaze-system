"""外部 MCP Server 工具运行时装载器测试。

用 conftest 的 db fixture（真实 MySQL 测试库 + SAVEPOINT 回滚）验证命名空间
反向关联与工具装载；会话连接（list_tools/call_tool）经实例方法 mock 规避联网。
"""

import pytest

from app.models.entity.sys_ai_mcp_namespace import SysAiMcpNamespace
from app.models.entity.sys_ai_mcp_server import SysAiMcpServer
from app.repository.ai_mcp_namespace_repository import ai_mcp_namespace_repository
from app.repository.ai_mcp_server_repository import ai_mcp_server_repository
from app.service.ai_mcp.mcp_external_tool_loader import (
    McpExternalToolLoader,
    _build_args_schema,
)

pytestmark = pytest.mark.requires_db

# 测试环境 DNS 无法解析外部域名，SSRF 守卫会保守拒绝；默认 mock 放行，
# SSRF 拦截语义由 test_mcp_connection 覆盖，本文件专注装载逻辑。
@pytest.fixture(autouse=True)
def _allow_ssrf(monkeypatch):
    async def _allow(_server):
        return True, ""

    monkeypatch.setattr(
        "app.service.ai_mcp.mcp_external_tool_loader.apply_ssrf_guard", _allow
    )


_TOOLS = [
    {
        "name": "dehaze",
        "description": "图像去雾",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "evaluate",
        "description": "质量评估",
        "input_schema": {"type": "object"},
    },
]


async def _create_server(db, *, name="ext_srv", endpoint="https://mcp.example.com/mcp", status=1):
    server = SysAiMcpServer(
        name=name,
        description="",
        protocol_type="streamable-http",
        endpoint=endpoint,
        status=status,
        tool_count=0,
    )
    await ai_mcp_server_repository.create(db, server)
    return server


async def _create_namespace(db, server_id, namespace, tool_names=None):
    await ai_mcp_namespace_repository.create(
        db,
        SysAiMcpNamespace(server_id=server_id, namespace=namespace, tool_names=tool_names),
    )


class TestArgsSchema:
    def test_required_and_optional_fields(self):
        schema = {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "图片地址"},
                "limit": {"type": "integer"},
            },
            "required": ["url"],
        }
        model = _build_args_schema(schema)
        assert model is not None
        fields = model.model_fields
        assert "url" in fields and "limit" in fields
        assert fields["url"].is_required() is True
        assert fields["limit"].is_required() is False

    def test_empty_properties_returns_none(self):
        assert _build_args_schema({"type": "object"}) is None


class TestLoadTools:
    async def test_empty_namespaces_returns_empty(self, db):
        loader = McpExternalToolLoader()
        assert await loader.load_tools(db, [], {}) == []

    async def test_no_enabled_servers_returns_empty(self, db):
        loader = McpExternalToolLoader()
        await _create_server(db, status=0)  # 禁用
        assert await loader.load_tools(db, ["image"], {}) == []

    @staticmethod
    async def _fake_list(_srv):
        return _TOOLS

    async def test_scoped_by_namespace_tool_names(self, db, monkeypatch):
        server = await _create_server(db)
        await _create_namespace(db, server.id, "image", tool_names=["dehaze"])
        loader = McpExternalToolLoader()
        monkeypatch.setattr(loader, "_list_tools", self._fake_list)

        tools = await loader.load_tools(db, ["image"], {})

        assert [t.name for t in tools] == ["image_dehaze"]
        assert "去雾" in tools[0].description

    async def test_all_tools_when_namespace_no_scope(self, db, monkeypatch):
        server = await _create_server(db)
        await _create_namespace(db, server.id, "image", tool_names=[])
        loader = McpExternalToolLoader()
        monkeypatch.setattr(loader, "_list_tools", self._fake_list)

        tools = await loader.load_tools(db, ["image"], {})
        assert {t.name for t in tools} == {"image_dehaze", "image_evaluate"}

    async def test_irrelevant_namespace_not_loaded(self, db, monkeypatch):
        server = await _create_server(db)
        await _create_namespace(db, server.id, "video", tool_names=[])
        loader = McpExternalToolLoader()
        monkeypatch.setattr(loader, "_list_tools", self._fake_list)

        assert await loader.load_tools(db, ["image"], {}) == []

    async def test_ssrf_guard_blocks_internal_endpoint(self, db, monkeypatch):
        server = await _create_server(db, endpoint="http://192.168.1.1/mcp")
        await _create_namespace(db, server.id, "image", tool_names=[])
        loader = McpExternalToolLoader()
        monkeypatch.setattr(loader, "_list_tools", self._fake_list)

        async def _deny(_server):
            return False, "MCP 端点不安全（仅允许 https 且禁止内网地址）"

        monkeypatch.setattr(
            "app.service.ai_mcp.mcp_external_tool_loader.apply_ssrf_guard", _deny
        )

        tools = await loader.load_tools(db, ["image"], {})
        assert tools == []


class TestToolCall:
    async def test_success_returns_text_and_records_audit(self, db, monkeypatch):
        server = await _create_server(db)
        loader = McpExternalToolLoader()
        recorded = {}

        async def _fake_call(_server, name, arguments):
            return f"result of {name}: {arguments}"

        async def _fake_record(
            ctx, server_id, server_name, tool_name, result, latency_ms, request, response
        ):
            recorded.update(locals())

        monkeypatch.setattr(
            "app.service.ai_mcp.mcp_external_tool_loader.call_remote_tool", _fake_call
        )
        monkeypatch.setattr(loader, "_record_call", _fake_record)

        tool = loader._build_tool(
            server, "image", _TOOLS[0], {"user_id": 7}
        )
        out = await tool.ainvoke({"url": "https://x/a.png"})

        assert out == "result of dehaze: {'url': 'https://x/a.png'}"
        assert recorded["result"] == "success"
        assert recorded["server_id"] == server.id
        assert recorded["tool_name"] == "dehaze"
        assert recorded["latency_ms"] >= 0

    async def test_failure_returns_error_text_and_records_failure(self, db, monkeypatch):
        server = await _create_server(db)
        loader = McpExternalToolLoader()
        recorded = {}

        async def _fake_call(_server, name, arguments):
            raise RuntimeError("downstream boom")

        async def _fake_record(
            ctx, server_id, server_name, tool_name, result, latency_ms, request, response
        ):
            recorded.update(locals())

        monkeypatch.setattr(
            "app.service.ai_mcp.mcp_external_tool_loader.call_remote_tool", _fake_call
        )
        monkeypatch.setattr(loader, "_record_call", _fake_record)

        tool = loader._build_tool(server, "image", _TOOLS[0], {})
        out = await tool.ainvoke({"url": "https://x/a.png"})

        assert "downstream boom" in out
        assert recorded["result"] == "failure"
