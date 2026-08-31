"""外部 MCP Server 注册管理服务

职责边界（与 mcp-toolkit 分工）：
- 本服务：Server 注册表 CRUD、凭据 AES 加密、健康探测、命名空间配置
- mcp-toolkit：工具清单拉取、调用审计、市场接入

安全：
- 凭据仅落 AES 密文（复用 aes_cipher.encrypt），日志不打印明文
- 健康探测目标仅 https 且禁内网（SSRF 防护），stdio 无端点可探测则返回离线
"""

from __future__ import annotations

import logging
import time

import httpx

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.crypto.aes_cipher import encrypt
from app.models.entity.sys_ai_mcp_namespace import SysAiMcpNamespace
from app.models.entity.sys_ai_mcp_server import SysAiMcpServer
from app.models.schema.ai_mcp import (
    McpCredentialForm,
    McpHealthResult,
    McpNamespaceItem,
    McpServerCreate,
    McpServerResult,
    McpServerUpdate,
)
from app.repository.ai_mcp_namespace_repository import ai_mcp_namespace_repository
from app.repository.ai_mcp_server_repository import ai_mcp_server_repository
from app.service.ai_mcp.mcp_connection import check_endpoint_safe

logger = logging.getLogger(__name__)

# 健康探测超时（秒）：探测失败降级离线标记，不阻断管理流程
_HEALTH_TIMEOUT = 5.0

# 支持网络探测的协议（stdio 为本地进程，无端点可探测）
_PROBEABLE_PROTOCOLS = frozenset({"streamable-http", "sse"})


class AiMcpServerService:
    # ── Server 注册表 CRUD ─────────────────────────────────

    async def create_server(self, db, form: McpServerCreate) -> McpServerResult:
        if await ai_mcp_server_repository.get_by_name(db, form.name, include_deleted=True):
            raise BusinessException(ResultCode.DATA_EXISTS, "MCP Server 名称已存在")
        server = SysAiMcpServer(
            name=form.name,
            description=form.description,
            protocol_type=form.protocol_type,
            endpoint=form.endpoint,
            auth_type=form.auth_type,
        )
        server = await ai_mcp_server_repository.create(db, server)
        return McpServerResult.model_validate(server)

    async def update_server(self, db, server_id: int, form: McpServerUpdate) -> McpServerResult:
        server = await self._get_server_or_raise(db, server_id)
        data = form.model_dump(exclude_unset=True)
        if "name" in data and data["name"]:
            existing = await ai_mcp_server_repository.get_by_name(
                db, data["name"], include_deleted=True
            )
            if existing and existing.id != server_id:
                raise BusinessException(ResultCode.DATA_EXISTS, "MCP Server 名称已存在")
        await ai_mcp_server_repository.update(db, server, data)
        return McpServerResult.model_validate(server)

    async def delete_server(self, db, server_id: int) -> None:
        server = await self._get_server_or_raise(db, server_id)
        await ai_mcp_server_repository.soft_delete_by_ids(db, [server.id])

    async def switch_server_status(self, db, server_id: int, status: int) -> McpServerResult:
        server = await self._get_server_or_raise(db, server_id)
        server.status = status
        await db.flush()
        await db.refresh(server)
        return McpServerResult.model_validate(server)

    async def list_servers(
        self,
        db,
        page: int,
        size: int,
        keyword: str | None = None,
        status: int | None = None,
    ) -> tuple[list[McpServerResult], int]:
        items, total = await ai_mcp_server_repository.paginate_servers(
            db, page, size, keyword, status
        )
        return [McpServerResult.model_validate(s) for s in items], total

    async def get_server(self, db, server_id: int) -> McpServerResult:
        return McpServerResult.model_validate(await self._get_server_or_raise(db, server_id))

    # ── 凭据（AES 加密存储，仅录入/更新，不回显明文） ───────────

    async def update_credentials(self, db, server_id: int, form: McpCredentialForm) -> None:
        server = await self._get_server_or_raise(db, server_id)
        payload = {}
        if form.api_key:
            payload["api_key"] = encrypt(form.api_key)
        if form.extra:
            payload["extra"] = {k: encrypt(v) for k, v in form.extra.items()}
        server.credentials = payload
        await db.flush()

    # ── 健康探测（失败降级离线标记，不阻断） ────────────────────

    async def probe_health(self, db, server_id: int) -> McpHealthResult:
        server = await self._get_server_or_raise(db, server_id)
        if server.protocol_type not in _PROBEABLE_PROTOCOLS or not server.endpoint:
            server.health = "offline"
            await db.flush()
            return McpHealthResult(status="offline")

        status, latency_ms = await self._probe(server)
        server.health = status
        await db.flush()
        return McpHealthResult(status=status, latency_ms=latency_ms)

    async def _probe(self, server: SysAiMcpServer) -> tuple[str, int | None]:
        """探测外部端点连通性，返回 (status, latency_ms)；异常一律离线降级。

        streamable-http 按 MCP 协议以 POST initialize（JSON-RPC 2.0）探测，
        其余协议回退 HTTP GET；端点经统一 SSRF 守卫（预设白名单放行）。
        """
        if not await check_endpoint_safe(server.endpoint):
            return "offline", None
        try:
            async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT, follow_redirects=True) as client:
                start = time.monotonic()
                if server.protocol_type == "streamable-http":
                    resp = await client.post(
                        server.endpoint,
                        headers={"Content-Type": "application/json"},
                        json={
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "initialize",
                            "params": {
                                "protocolVersion": "2025-03-26",
                                "capabilities": {},
                                "clientInfo": {"name": "dehaze-health-probe", "version": "1.0"},
                            },
                        },
                    )
                else:
                    resp = await client.get(server.endpoint)
                latency_ms = int((time.monotonic() - start) * 1000)
            return ("online" if resp.status_code < 400 else "offline"), latency_ms
        except (httpx.HTTPError, Exception) as exc:  # noqa: BLE001 - 探测失败不阻断管理流程
            logger.warning("MCP Server %s 健康探测失败: %s", server.id, exc)
            return "offline", None

    # ── 命名空间（覆盖式更新） ──────────────────────────────

    async def list_namespaces(self, db, server_id: int) -> list[McpNamespaceItem]:
        await self._get_server_or_raise(db, server_id)
        rows = await ai_mcp_namespace_repository.list_by_server(db, server_id)
        return [McpNamespaceItem(name=r.namespace, toolNames=r.tool_names or []) for r in rows]

    async def update_namespaces(
        self, db, server_id: int, items: list[McpNamespaceItem]
    ) -> list[McpNamespaceItem]:
        await self._get_server_or_raise(db, server_id)
        await ai_mcp_namespace_repository.delete_by_server(db, server_id)
        for item in items:
            await ai_mcp_namespace_repository.create(
                db,
                SysAiMcpNamespace(
                    server_id=server_id, namespace=item.name, tool_names=item.toolNames
                ),
            )
        await db.flush()
        return await self.list_namespaces(db, server_id)

    # ── 工具 ──────────────────────────────────────────────

    async def _get_server_or_raise(self, db, server_id: int) -> SysAiMcpServer:
        server = await ai_mcp_server_repository.get_by_id(db, server_id)
        if not server:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "MCP Server 不存在")
        return server


ai_mcp_server_service = AiMcpServerService()
