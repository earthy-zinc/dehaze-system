"""外部 MCP Server 服务层测试（真实 db fixture + respx 健康探测）"""

import re

import httpx
import pytest
import respx
from pydantic import ValidationError

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.crypto.aes_cipher import decrypt
from app.models.entity.sys_ai_agent_mcp import SysAiAgentMcp
from app.models.entity.sys_ai_mcp_namespace import SysAiMcpNamespace
from app.models.entity.sys_ai_mcp_tool import SysAiMcpTool
from app.models.schema.ai_mcp import (
    McpCredentialForm,
    McpNamespaceItem,
    McpServerCreate,
    McpServerUpdate,
)
from app.repository.ai_mcp_namespace_repository import ai_mcp_namespace_repository
from app.repository.ai_mcp_server_repository import ai_mcp_server_repository
from app.repository.ai_mcp_tool_repository import ai_mcp_tool_repository
from app.service.ai_mcp.ai_mcp_server_service import ai_mcp_server_service
from app.service.ai_mcp.mcp_health_checker import mcp_health_checker

pytestmark = pytest.mark.requires_db

_PROBE_URL_RE = re.compile(r"https://example\.com/mcp")


async def _probe_safe(_url: str) -> bool:
    """桩掉 check_endpoint_safe 的真实 DNS 解析（本机 example.com 可能被解析到回环地址）。"""
    return True


def _server_form(**overrides) -> McpServerCreate:
    data: dict = {
        "name": "test_mcp_server",
        "description": "测试 Server",
        "protocol_type": "streamable-http",
        "endpoint": "https://example.com/mcp",
        "auth_type": "api_key",
    }
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


async def test_create_server_rejects_stdio_protocol():
    """stdio 无网络端点，拉取/装载/探测三处均不支持，注册入口直接拒绝（防死条目）。"""
    with pytest.raises(ValidationError):
        _server_form(protocol_type="stdio")


async def test_update_server_rejects_stdio_protocol():
    with pytest.raises(ValidationError):
        # 负向用例：故意构造非法值
        McpServerUpdate(protocol_type="stdio")  # pyright: ignore[reportArgumentType]


async def test_create_server_rejects_empty_endpoint(db, mock_redis):
    with pytest.raises(BusinessException) as exc:
        await ai_mcp_server_service.create_server(db, _server_form(endpoint=None))
    assert exc.value.code == ResultCode.PARAM_ERROR


async def test_update_server_rejects_empty_endpoint(db, mock_redis):
    created = await ai_mcp_server_service.create_server(db, _server_form())
    with pytest.raises(BusinessException) as exc:
        await ai_mcp_server_service.update_server(db, created.id, McpServerUpdate(endpoint=""))
    assert exc.value.code == ResultCode.PARAM_ERROR


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
    # 软删标记（deleted 写为行 id，保留软删历史；同名活跃查重可重建）
    row = await ai_mcp_server_repository.get_by_id(db, created.id, with_deleted=True)
    assert row is not None
    assert row.deleted != 0


async def test_delete_server_blocked_when_agent_references_namespace(db, mock_redis):
    """删除前校验 Agent 关联：Agent 关联了该 Server 的命名空间时拒绝删除。"""
    created = await ai_mcp_server_service.create_server(db, _server_form())
    await ai_mcp_namespace_repository.create(
        db, SysAiMcpNamespace(server_id=created.id, namespace="image_processing", tool_names=[])
    )
    db.add(SysAiAgentMcp(agent_id=5, mcp_namespace="image_processing"))
    await db.flush()

    with pytest.raises(BusinessException) as exc:
        await ai_mcp_server_service.delete_server(db, created.id)
    assert exc.value.code == ResultCode.DATA_BIND_EXISTS
    # 未被删除
    assert await ai_mcp_server_repository.get_by_id(db, created.id) is not None


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
    assert row is not None
    assert row.credentials is not None
    assert row.credentials["api_key"] != "secret_key_123"
    assert decrypt(row.credentials["api_key"]) == "secret_key_123"
    assert decrypt(row.credentials["extra"]["token"]) == "extra_secret"


async def test_update_credentials_merge_keeps_api_key(db, mock_redis):
    """合并式更新：仅提交 extra 时 api_key 保留原值（对齐前端"留空不更新"语义）。"""
    created = await ai_mcp_server_service.create_server(db, _server_form())
    await ai_mcp_server_service.update_credentials(
        db, created.id, McpCredentialForm(api_key="keep_me_key")
    )
    await ai_mcp_server_service.update_credentials(
        db, created.id, McpCredentialForm(extra={"oauth_token": "tok2"})
    )
    row = await ai_mcp_server_repository.get_by_id(db, created.id)
    assert row is not None
    assert row.credentials is not None
    assert decrypt(row.credentials["api_key"]) == "keep_me_key"
    assert decrypt(row.credentials["extra"]["oauth_token"]) == "tok2"


async def test_update_credentials_empty_form_noop(db, mock_redis):
    created = await ai_mcp_server_service.create_server(db, _server_form())
    await ai_mcp_server_service.update_credentials(
        db, created.id, McpCredentialForm(api_key="k1", extra={"a": "1"})
    )
    await ai_mcp_server_service.update_credentials(db, created.id, McpCredentialForm())
    row = await ai_mcp_server_repository.get_by_id(db, created.id)
    assert row is not None
    assert row.credentials is not None
    assert decrypt(row.credentials["api_key"]) == "k1"
    assert decrypt(row.credentials["extra"]["a"]) == "1"


async def test_update_credentials_clear_removes_all(db, mock_redis):
    """凭据轮换/吊销：clear=true 整体清空，credential_configured 归 false。"""
    created = await ai_mcp_server_service.create_server(db, _server_form())
    await ai_mcp_server_service.update_credentials(
        db, created.id, McpCredentialForm(api_key="k1", extra={"a": "1"})
    )
    await ai_mcp_server_service.update_credentials(db, created.id, McpCredentialForm(clear=True))

    row = await ai_mcp_server_repository.get_by_id(db, created.id)
    assert row is not None
    assert row.credentials is None
    assert row.credential_configured is False


async def test_update_credentials_clear_then_set_again(db, mock_redis):
    """清除后重新录入：轮换路径可闭环（不因历史密文残留而合并）。"""
    created = await ai_mcp_server_service.create_server(db, _server_form())
    await ai_mcp_server_service.update_credentials(
        db, created.id, McpCredentialForm(api_key="old_key", extra={"a": "1"})
    )
    await ai_mcp_server_service.update_credentials(db, created.id, McpCredentialForm(clear=True))
    await ai_mcp_server_service.update_credentials(
        db, created.id, McpCredentialForm(api_key="new_key")
    )

    row = await ai_mcp_server_repository.get_by_id(db, created.id)
    assert row is not None
    assert row.credentials is not None
    assert decrypt(row.credentials["api_key"]) == "new_key"
    assert "extra" not in row.credentials


async def _seed_tools(db, server_id: int, *names: str) -> None:
    """预置已拉取到的工具清单（命名空间工具名校验的基准）。"""
    for name in names:
        await ai_mcp_tool_repository.create(db, SysAiMcpTool(server_id=server_id, name=name))


async def test_update_namespaces_overwrite(db, mock_redis):
    created = await ai_mcp_server_service.create_server(db, _server_form())
    await _seed_tools(db, created.id, "tool_a", "tool_b", "tool_c")
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


async def test_update_namespaces_rejects_unknown_tool(db, mock_redis):
    """拼错/未拉取到的工具名静默失效，须拒绝并提示可用工具清单。"""
    created = await ai_mcp_server_service.create_server(db, _server_form())
    await _seed_tools(db, created.id, "tool_a", "tool_b")

    with pytest.raises(BusinessException) as exc:
        await ai_mcp_server_service.update_namespaces(
            db, created.id, [McpNamespaceItem(name="ns_a", toolNames=["tool_a", "tool_typo"])]
        )
    assert exc.value.code == ResultCode.PARAM_ERROR
    assert "tool_typo" in exc.value.message
    assert "tool_a" in exc.value.message
    # 拒绝后不落库
    assert await ai_mcp_server_service.list_namespaces(db, created.id) == []


async def test_update_namespaces_rejects_all_when_tools_not_fetched(db, mock_redis):
    """未拉取工具清单时任何具名工具都被拒（提示先拉取），避免凭空写入死配置。"""
    created = await ai_mcp_server_service.create_server(db, _server_form())

    with pytest.raises(BusinessException) as exc:
        await ai_mcp_server_service.update_namespaces(
            db, created.id, [McpNamespaceItem(name="ns_a", toolNames=["tool_a"])]
        )
    assert exc.value.code == ResultCode.PARAM_ERROR
    assert "拉取" in exc.value.message


async def test_list_namespaces(db, mock_redis):
    created = await ai_mcp_server_service.create_server(db, _server_form())
    await _seed_tools(db, created.id, "tool_a")
    await ai_mcp_server_service.update_namespaces(
        db, created.id, [McpNamespaceItem(name="ns_a", toolNames=["tool_a"])]
    )
    namespaces = await ai_mcp_server_service.list_namespaces(db, created.id)
    assert namespaces == [McpNamespaceItem(name="ns_a", toolNames=["tool_a"])]


async def test_update_namespaces_rejects_illegal_name(db, mock_redis):
    """对抗语料：零宽字符/CRLF/中文/空格/超长命名空间名均拒绝（工具运行时名须合法）。"""
    created = await ai_mcp_server_service.create_server(db, _server_form())
    for bad in [
        "ns\u200b_zwsp",  # 零宽字符
        "ns\r\n_inject",  # CRLF
        "ns 中文",  # 中文与空格
        "1_starts_with_digit",  # 数字开头
        "ns" * 40,  # 超长（>64）
    ]:
        with pytest.raises(BusinessException) as exc:
            await ai_mcp_server_service.update_namespaces(
                db, created.id, [McpNamespaceItem(name=bad, toolNames=[])]
            )
        assert exc.value.code == ResultCode.PARAM_ERROR


async def test_update_namespaces_rejects_duplicate_and_empty_tool(db, mock_redis):
    created = await ai_mcp_server_service.create_server(db, _server_form())
    with pytest.raises(BusinessException) as exc:
        await ai_mcp_server_service.update_namespaces(
            db,
            created.id,
            [
                McpNamespaceItem(name="ns_a", toolNames=[]),
                McpNamespaceItem(name="ns_a", toolNames=[]),
            ],
        )
    assert exc.value.code == ResultCode.PARAM_ERROR
    with pytest.raises(BusinessException) as exc:
        await ai_mcp_server_service.update_namespaces(
            db, created.id, [McpNamespaceItem(name="ns_a", toolNames=["ok", ""])]
        )
    assert exc.value.code == ResultCode.PARAM_ERROR


async def test_probe_health_online(db, mock_redis, monkeypatch):
    monkeypatch.setattr("app.service.ai_mcp.ai_mcp_server_service.check_endpoint_safe", _probe_safe)
    created = await ai_mcp_server_service.create_server(db, _server_form())
    with respx.mock(assert_all_mocked=True) as router:
        router.post(_PROBE_URL_RE).mock(return_value=httpx.Response(200))
        health = await ai_mcp_server_service.probe_health(db, created.id)
    assert health.status == "online"
    assert health.latency_ms is not None


async def test_probe_health_http_error_offline(db, mock_redis, monkeypatch):
    monkeypatch.setattr("app.service.ai_mcp.ai_mcp_server_service.check_endpoint_safe", _probe_safe)
    created = await ai_mcp_server_service.create_server(db, _server_form())
    with respx.mock(assert_all_mocked=True) as router:
        router.post(_PROBE_URL_RE).mock(return_value=httpx.Response(500))
        health = await ai_mcp_server_service.probe_health(db, created.id)
    assert health.status == "offline"


async def test_probe_health_timeout_offline(db, mock_redis, monkeypatch):
    monkeypatch.setattr("app.service.ai_mcp.ai_mcp_server_service.check_endpoint_safe", _probe_safe)
    created = await ai_mcp_server_service.create_server(db, _server_form())
    with respx.mock(assert_all_mocked=True) as router:
        router.post(_PROBE_URL_RE).mock(side_effect=httpx.ConnectTimeout("timeout"))
        health = await ai_mcp_server_service.probe_health(db, created.id)
    assert health.status == "offline"


async def test_probe_health_carries_credentials(db, mock_redis, monkeypatch):
    """需鉴权的 Server 裸探测恒被 401 判离线，探测须带凭据头。"""
    monkeypatch.setattr("app.service.ai_mcp.ai_mcp_server_service.check_endpoint_safe", _probe_safe)
    created = await ai_mcp_server_service.create_server(db, _server_form())
    await ai_mcp_server_service.update_credentials(
        db, created.id, McpCredentialForm(api_key="probe_key_123")
    )
    with respx.mock(assert_all_mocked=True) as router:
        route = router.post(_PROBE_URL_RE).mock(return_value=httpx.Response(200))
        health = await ai_mcp_server_service.probe_health(db, created.id)
    assert health.status == "online"
    assert route.calls[0].request.headers["Authorization"] == "Bearer probe_key_123"


async def test_probe_health_sse_carries_credentials(db, mock_redis, monkeypatch):
    monkeypatch.setattr("app.service.ai_mcp.ai_mcp_server_service.check_endpoint_safe", _probe_safe)
    created = await ai_mcp_server_service.create_server(db, _server_form(protocol_type="sse"))
    await ai_mcp_server_service.update_credentials(
        db, created.id, McpCredentialForm(api_key="sse_key_456")
    )
    with respx.mock(assert_all_mocked=True) as router:
        route = router.get(_PROBE_URL_RE).mock(return_value=httpx.Response(200))
        health = await ai_mcp_server_service.probe_health(db, created.id)
    assert health.status == "online"
    assert route.calls[0].request.headers["Authorization"] == "Bearer sse_key_456"


async def test_probe_health_records_last_check_time(db, mock_redis, monkeypatch):
    monkeypatch.setattr("app.service.ai_mcp.ai_mcp_server_service.check_endpoint_safe", _probe_safe)
    created = await ai_mcp_server_service.create_server(db, _server_form())
    row = await ai_mcp_server_repository.get_by_id(db, created.id)
    assert row is not None
    assert row.last_check_time is None

    with respx.mock(assert_all_mocked=True) as router:
        router.post(_PROBE_URL_RE).mock(return_value=httpx.Response(200))
        await ai_mcp_server_service.probe_health(db, created.id)

    row = await ai_mcp_server_repository.get_by_id(db, created.id)
    assert row is not None
    assert row.last_check_time is not None


async def test_probe_health_does_not_follow_redirect(db, mock_redis, monkeypatch):
    """SSRF：探测不跟随重定向，30x 跳内网目标不发起请求（KB 轮同口径）。"""
    monkeypatch.setattr("app.service.ai_mcp.ai_mcp_server_service.check_endpoint_safe", _probe_safe)
    created = await ai_mcp_server_service.create_server(db, _server_form())
    # assert_all_called=False：跳转目标路由预期零调用（断言在前置条件中显式做出）
    with respx.mock(assert_all_mocked=False, assert_all_called=False) as router:
        router.post(_PROBE_URL_RE).mock(
            return_value=httpx.Response(
                302, headers={"Location": "http://169.254.169.254/latest/meta-data"}
            )
        )
        redirect_target = router.get("http://169.254.169.254/latest/meta-data").mock(
            return_value=httpx.Response(200)
        )
        health = await ai_mcp_server_service.probe_health(db, created.id)
    assert redirect_target.call_count == 0
    assert health.status == "offline"


async def test_get_server_not_found(db, mock_redis):
    with pytest.raises(BusinessException) as exc:
        await ai_mcp_server_service.get_server(db, 9999)
    assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND


async def _graph_cache_cleared_after_commit(db, mutate) -> bool:
    """执行变更操作并触发提交后回调，返回图缓存是否被清空。

    提交前不清（回滚场景不应丢缓存），提交后必须清——缓存的图在构建时装载了
    外部工具，Server 启停/配置变更后旧图仍持旧工具集与旧凭据。
    """
    from app.database import run_after_commit_callbacks
    from app.service.ai.service.reasoning_service import reasoning_service

    reasoning_service._graphs[(999, 0, "")] = object()
    await mutate()
    assert reasoning_service._graphs, "事务未提交前不应清图缓存"
    await run_after_commit_callbacks(db)
    cleared = not reasoning_service._graphs
    reasoning_service._graphs.clear()
    return cleared


class TestGraphCacheInvalidation:
    async def test_switch_status_clears_graph_cache(self, db, mock_redis):
        created = await ai_mcp_server_service.create_server(db, _server_form())
        cleared = await _graph_cache_cleared_after_commit(
            db, lambda: ai_mcp_server_service.switch_server_status(db, created.id, 0)
        )
        assert cleared is True

    async def test_update_server_clears_graph_cache(self, db, mock_redis):
        created = await ai_mcp_server_service.create_server(db, _server_form())
        cleared = await _graph_cache_cleared_after_commit(
            db,
            lambda: ai_mcp_server_service.update_server(
                db, created.id, McpServerUpdate(endpoint="https://example.org/mcp")
            ),
        )
        assert cleared is True

    async def test_credentials_rotation_clears_graph_cache(self, db, mock_redis):
        created = await ai_mcp_server_service.create_server(db, _server_form())
        cleared = await _graph_cache_cleared_after_commit(
            db,
            lambda: ai_mcp_server_service.update_credentials(
                db, created.id, McpCredentialForm(api_key="rotated_key")
            ),
        )
        assert cleared is True

    async def test_namespace_change_clears_graph_cache(self, db, mock_redis):
        created = await ai_mcp_server_service.create_server(db, _server_form())
        await _seed_tools(db, created.id, "tool_a")
        cleared = await _graph_cache_cleared_after_commit(
            db,
            lambda: ai_mcp_server_service.update_namespaces(
                db, created.id, [McpNamespaceItem(name="ns_a", toolNames=["tool_a"])]
            ),
        )
        assert cleared is True

    async def test_delete_clears_graph_cache(self, db, mock_redis):
        created = await ai_mcp_server_service.create_server(db, _server_form())
        cleared = await _graph_cache_cleared_after_commit(
            db, lambda: ai_mcp_server_service.delete_server(db, created.id)
        )
        assert cleared is True

    async def test_change_broadcasts_to_other_instances(self, db, mock_redis, monkeypatch):
        """变更后须广播失效：图缓存是进程内的，除本实例外其余实例各自失效。

        java/go 原生改 MCP（共享库）时本端没有本地写操作，只能靠该广播失效。
        """
        broadcast: list[bool] = []

        async def _spy() -> None:
            broadcast.append(True)

        monkeypatch.setattr("app.infrastructure.cache.cache.publish_graph_invalidation", _spy)
        created = await ai_mcp_server_service.create_server(db, _server_form())
        await _graph_cache_cleared_after_commit(
            db, lambda: ai_mcp_server_service.switch_server_status(db, created.id, 0)
        )
        assert broadcast == [True]


class TestHealthChecker:
    async def test_refreshes_enabled_servers_only(self, db, mock_redis, monkeypatch):
        """巡检刷新启用中 Server 的 health/last_check_time；禁用 Server 不参与探测。"""
        monkeypatch.setattr(
            "app.service.ai_mcp.ai_mcp_server_service.check_endpoint_safe", _probe_safe
        )
        enabled = await ai_mcp_server_service.create_server(db, _server_form(name="probe_on"))
        disabled = await ai_mcp_server_service.create_server(db, _server_form(name="probe_off"))
        await ai_mcp_server_service.switch_server_status(db, disabled.id, 0)

        with respx.mock(assert_all_mocked=True) as router:
            router.post(_PROBE_URL_RE).mock(return_value=httpx.Response(200))
            await mcp_health_checker.check_all(db)

        on_row = await ai_mcp_server_repository.get_by_id(db, enabled.id)
        assert on_row is not None
        assert on_row.health == "online"
        assert on_row.last_check_time is not None
        off_row = await ai_mcp_server_repository.get_by_id(db, disabled.id)
        assert off_row is not None
        assert off_row.health is None
        assert off_row.last_check_time is None

    async def test_probe_failure_marks_offline_without_raising(self, db, mock_redis, monkeypatch):
        monkeypatch.setattr(
            "app.service.ai_mcp.ai_mcp_server_service.check_endpoint_safe", _probe_safe
        )
        created = await ai_mcp_server_service.create_server(db, _server_form())

        with respx.mock(assert_all_mocked=True) as router:
            router.post(_PROBE_URL_RE).mock(side_effect=httpx.ConnectTimeout("timeout"))
            await mcp_health_checker.check_all(db)

        row = await ai_mcp_server_repository.get_by_id(db, created.id)
        assert row is not None
        assert row.health == "offline"
        assert row.last_check_time is not None
