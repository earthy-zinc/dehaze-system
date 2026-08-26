"""外部 MCP Server 工具/命名空间/市场/调用审计服务单元测试。

用 conftest 的 db fixture（真实 MySQL 测试库 + SAVEPOINT 回滚）验证真实落库行为；
工具拉取通过构造注入 mock fetcher 规避真实联网（不触达远端 MCP Server）。
"""

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_mcp_server import SysAiMcpServer
from app.models.schema.ai_mcp import McpCallQuery
from app.repository.ai_mcp_server_repository import ai_mcp_server_repository
from app.repository.ai_mcp_tool_repository import ai_mcp_tool_repository
from app.service.ai_mcp.mcp_manage_service import McpManageService


class _FakeFetcher:
    """mock 工具拉取器：按需返回工具列表或抛错，验证服务层落库逻辑。"""

    def __init__(self, tools=None, error=None):
        self.tools = tools or []
        self.error = error

    async def list_tools(self, server):
        if self.error:
            raise self.error
        return self.tools


@pytest.fixture
def service(fetcher):
    return McpManageService(fetcher=fetcher)


@pytest.fixture
def fetcher():
    return _FakeFetcher()


async def _create_server(db, name="test_mcp_srv"):
    server = SysAiMcpServer(
        name=name,
        description="测试",
        protocol_type="streamable-http",
        endpoint="https://example.com/mcp",
        auth_type="api_key",
        status=1,
        tool_count=0,
    )
    await ai_mcp_server_repository.create(db, server)
    return server


pytestmark = pytest.mark.requires_db


class TestGetTools:
    async def test_fetches_and_persists_tools(self, db, fetcher):
        fetcher.tools = [
            {"name": "image_dehaze", "description": "去雾", "input_schema": {"type": "object"}},
            {"name": "image_evaluate", "description": "评估", "input_schema": None},
        ]
        server = await _create_server(db)

        tools = await McpManageService(fetcher=fetcher).get_tools(db, server.id)

        assert [t.name for t in tools] == ["image_dehaze", "image_evaluate"]
        assert tools[0].description == "去雾"
        assert tools[0].input_schema == {"type": "object"}
        persisted = await ai_mcp_tool_repository.list_by_server(db, server.id)
        assert len(persisted) == 2
        assert (await ai_mcp_server_repository.get_by_id(db, server.id)).tool_count == 2

    async def test_replaces_previous_tools(self, db, fetcher):
        server = await _create_server(db)
        fetcher.tools = [{"name": "old_tool", "description": "旧", "input_schema": {}}]
        svc = McpManageService(fetcher=fetcher)
        await svc.get_tools(db, server.id)

        fetcher.tools = [{"name": "new_tool", "description": "新", "input_schema": {}}]
        tools = await svc.get_tools(db, server.id)

        assert [t.name for t in tools] == ["new_tool"]
        persisted = await ai_mcp_tool_repository.list_by_server(db, server.id)
        assert [t.name for t in persisted] == ["new_tool"]

    async def test_fetcher_failure_yields_empty_tools(self, db, fetcher):
        fetcher.error = ConnectionError("网络不可达")
        server = await _create_server(db)

        tools = await McpManageService(fetcher=fetcher).get_tools(db, server.id)

        assert tools == []
        assert (await ai_mcp_server_repository.get_by_id(db, server.id)).tool_count == 0

    async def test_server_not_found_raises(self, db, fetcher):
        with pytest.raises(BusinessException) as exc:
            await McpManageService(fetcher=fetcher).get_tools(db, 99999)
        assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND


class TestMarket:
    async def test_not_installed_when_no_server(self, db, fetcher):
        market = await McpManageService(fetcher=fetcher).get_market(db)
        assert all(not p.installed for p in market)
        assert {p.preset_id for p in market} == {"github", "mysql", "search"}

    async def test_installed_when_same_name_server_exists(self, db, fetcher):
        await _create_server(db, name="GitHub")
        market = await McpManageService(fetcher=fetcher).get_market(db)
        github = next(p for p in market if p.preset_id == "github")
        assert github.installed is True

    async def test_install_creates_server_and_fetches_tools(self, db, fetcher):
        fetcher.tools = [{"name": "repo_list", "description": "仓库列表", "input_schema": {}}]
        server = await McpManageService(fetcher=fetcher).install_preset(db, object(), "github")

        assert server.name == "GitHub"
        assert server.tool_count == 1
        persisted = await ai_mcp_tool_repository.list_by_server(db, server.id)
        assert [t.name for t in persisted] == ["repo_list"]

    async def test_install_reuses_existing_server(self, db, fetcher):
        existing = await _create_server(db, name="GitHub")
        server = await McpManageService(fetcher=fetcher).install_preset(db, object(), "github")
        assert server.id == existing.id

    async def test_install_unknown_preset_raises(self, db, fetcher):
        with pytest.raises(BusinessException) as exc:
            await McpManageService(fetcher=fetcher).install_preset(db, object(), "unknown")
        assert exc.value.code == ResultCode.PARAM_ERROR


class TestCallAudit:
    async def test_record_success(self, db, fetcher):
        server = await _create_server(db)
        svc = McpManageService(fetcher=fetcher)

        await svc.record_call(
            db,
            user_id=1,
            server_id=server.id,
            server_name=server.name,
            tool_name="image_dehaze",
            result="success",
            latency_ms=120,
            request={"k": "v"},
            response="ok",
        )
        await db.flush()

        calls = await svc.list_calls(db, McpCallQuery())
        assert len(calls.list) == 1
        call = calls.list[0]
        assert call.result == "success"
        assert call.user_id == 1
        assert call.server_name == server.name
        assert call.tool_name == "image_dehaze"
        assert call.latency_ms == 120

    async def test_record_failure_persists_status(self, db, fetcher):
        server = await _create_server(db)
        svc = McpManageService(fetcher=fetcher)

        await svc.record_call(
            db,
            user_id=2,
            server_id=server.id,
            server_name=server.name,
            tool_name="image_dehaze",
            result="failure",
            latency_ms=50,
        )
        await db.flush()

        calls = await svc.list_calls(db, McpCallQuery(serverId=server.id))
        assert len(calls.list) == 1
        assert calls.list[0].result == "failure"

    async def test_list_calls_paginated_and_filtered(self, db, fetcher):
        server = await _create_server(db)
        svc = McpManageService(fetcher=fetcher)
        for i in range(5):
            await svc.record_call(
                db,
                user_id=1,
                server_id=server.id,
                server_name=server.name,
                tool_name=f"tool_{i % 2}",
                result="success",
            )
        await db.flush()

        page = await svc.list_calls(db, McpCallQuery(pageNum=1, pageSize=2))
        assert page.total == 5
        assert len(page.list) == 2

        filtered = await svc.list_calls(db, McpCallQuery(tool_name="tool_1"))
        assert all(c.tool_name == "tool_1" for c in filtered.list)
