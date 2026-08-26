"""外部 MCP Server 服务层测试（真实 db fixture + respx 健康探测）"""

import re

import httpx
import pytest
import respx

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.crypto.aes_cipher import decrypt
from app.models.schema.ai_mcp import (
    McpCredentialForm,
    McpNamespaceItem,
    McpServerCreate,
    McpServerUpdate,
)
from app.repository.ai_mcp_server_repository import ai_mcp_server_repository
from app.service.ai_mcp.ai_mcp_server_service import ai_mcp_server_service

pytestmark = pytest.mark.requires_db

_PROBE_URL_RE = re.compile(r"https://example\.com/mcp")


def _server_form(**overrides) -> McpServerCreate:
    data = dict(
        name="test_mcp_server",
        description="测试 Server",
        protocol_type="streamable-http",
        endpoint="https://example.com/mcp",
        auth_type="api_key",
    )
    data.update(overrides)
    return McpServerCreate(**data)


async def test_create_server_persists_fields(db, mock_redis):
    result = await ai_mcp_server_service.create_server(db, _server_form())
    assert result.id > 0
    assert result.name == "test_mcp_server"
    assert result.protocol_type == "streamable-http"
    assert result.endpoint == "https://example.com/mcp"
    assert result.status == 1
    assert result.tool_count == 0


async def test_create_server_duplicate_name(db, mock_redis):
    await ai_mcp_server_service.create_server(db, _server_form())
    with pytest.raises(BusinessException) as exc:
        await ai_mcp_server_service.create_server(db, _server_form())
    assert exc.value.code == ResultCode.DATA_EXISTS


async def test_update_server_description(db, mock_redis):
    created = await ai_mcp_server_service.create_server(db, _server_form())
    updated = await ai_mcp_server_service.update_server(
        db,
        created.id,
        McpServerUpdate(description="updated-desc"),
    )
    assert updated.id == created.id
    assert updated.description == "updated-desc"


async def test_update_server_rename_conflict(db, mock_redis):
    a = await ai_mcp_server_service.create_server(db, _server_form(name="mcp_a"))
    await ai_mcp_server_service.create_server(db, _server_form(name="mcp_b"))
    with pytest.raises(BusinessException) as exc:
        await ai_mcp_server_service.update_server(db, a.id, McpServerUpdate(name="mcp_b"))
    assert exc.value.code == ResultCode.DATA_EXISTS


async def test_switch_server_status(db, mock_redis):
    created = await ai_mcp_server_service.create_server(db, _server_form())
    disabled = await ai_mcp_server_service.switch_server_status(db, created.id, 0)
    assert disabled.status == 0
    enabled = await ai_mcp_server_service.switch_server_status(db, created.id, 1)
    assert enabled.status == 1


async def test_delete_server_soft_delete(db, mock_redis):
    created = await ai_mcp_server_service.create_server(db, _server_form())
    await ai_mcp_server_service.delete_server(db, created.id)
    # 软删标记 deleted=1（保留软删历史，同名查重含软删记录）
    row = await ai_mcp_server_repository.get_by_id(db, created.id, with_deleted=True)
    assert row is not None and row.deleted == 1


async def test_list_servers_pagination(db, mock_redis):
    for i in range(3):
        await ai_mcp_server_service.create_server(db, _server_form(name=f"mcp_list_{i}"))
    items, total = await ai_mcp_server_service.list_servers(db, page=1, size=2)
    assert total == 3
    assert len(items) == 2


async def test_update_credentials_encrypted(db, mock_redis):
    created = await ai_mcp_server_service.create_server(db, _server_form())
    await ai_mcp_server_service.update_credentials(
        db,
        created.id,
        McpCredentialForm(api_key="secret_key_123", extra={"token": "extra_secret"}),
    )
    row = await ai_mcp_server_repository.get_by_id(db, created.id)
    assert row.credentials["api_key"] != "secret_key_123"
    assert decrypt(row.credentials["api_key"]) == "secret_key_123"
    assert decrypt(row.credentials["extra"]["token"]) == "extra_secret"


async def test_update_namespaces_overwrite(db, mock_redis):
    created = await ai_mcp_server_service.create_server(db, _server_form())
    namespaces = await ai_mcp_server_service.update_namespaces(
        db,
        created.id,
        [McpNamespaceItem(name="ns_a", toolNames=["tool_a", "tool_b"])],
    )
    assert namespaces == [McpNamespaceItem(name="ns_a", toolNames=["tool_a", "tool_b"])]
    # 覆盖式：再次更新后仅保留新配置
    namespaces = await ai_mcp_server_service.update_namespaces(
        db, created.id, [McpNamespaceItem(name="ns_b", toolNames=["tool_c"])]
    )
    assert namespaces == [McpNamespaceItem(name="ns_b", toolNames=["tool_c"])]


async def test_list_namespaces(db, mock_redis):
    created = await ai_mcp_server_service.create_server(db, _server_form())
    await ai_mcp_server_service.update_namespaces(
        db, created.id, [McpNamespaceItem(name="ns_a", toolNames=["tool_a"])]
    )
    namespaces = await ai_mcp_server_service.list_namespaces(db, created.id)
    assert namespaces == [McpNamespaceItem(name="ns_a", toolNames=["tool_a"])]


async def test_probe_health_online(db, mock_redis):
    created = await ai_mcp_server_service.create_server(db, _server_form())
    with respx.mock(assert_all_mocked=True) as router:
        router.get(_PROBE_URL_RE).mock(return_value=httpx.Response(200))
        health = await ai_mcp_server_service.probe_health(db, created.id)
    assert health.status == "online"
    assert health.latency_ms is not None


async def test_probe_health_http_error_offline(db, mock_redis):
    created = await ai_mcp_server_service.create_server(db, _server_form())
    with respx.mock(assert_all_mocked=True) as router:
        router.get(_PROBE_URL_RE).mock(return_value=httpx.Response(500))
        health = await ai_mcp_server_service.probe_health(db, created.id)
    assert health.status == "offline"


async def test_probe_health_timeout_offline(db, mock_redis):
    created = await ai_mcp_server_service.create_server(db, _server_form())
    with respx.mock(assert_all_mocked=True) as router:
        router.get(_PROBE_URL_RE).mock(side_effect=httpx.ConnectTimeout("timeout"))
        health = await ai_mcp_server_service.probe_health(db, created.id)
    assert health.status == "offline"


async def test_probe_health_stdio_offline_without_probe(db, mock_redis):
    created = await ai_mcp_server_service.create_server(
        db, _server_form(protocol_type="stdio", endpoint=None)
    )
    health = await ai_mcp_server_service.probe_health(db, created.id)
    assert health.status == "offline"


async def test_get_server_not_found(db, mock_redis):
    with pytest.raises(BusinessException) as exc:
        await ai_mcp_server_service.get_server(db, 9999)
    assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND
