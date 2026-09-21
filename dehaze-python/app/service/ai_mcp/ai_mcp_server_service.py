"""外部 MCP Server 注册管理服务

职责边界（与 mcp-toolkit 分工）：
- 本服务：Server 注册表 CRUD、凭据 AES 加密、健康探测、命名空间配置
- mcp-toolkit：工具清单拉取、调用审计、市场接入

安全：
- 凭据仅落 AES 密文（复用 aes_cipher.encrypt），日志不打印明文
- 健康探测目标仅 https 且禁内网（SSRF 防护），携带 Server 凭据鉴权头
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime

import httpx
from sqlalchemy import func, select

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import defer_after_commit
from app.infrastructure.crypto.aes_cipher import encrypt
from app.models.entity.sys_ai_agent_mcp import SysAiAgentMcp
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
from app.repository.ai_mcp_tool_repository import ai_mcp_tool_repository
from app.service.ai_mcp.mcp_connection import build_mcp_auth_headers, check_endpoint_safe

logger = logging.getLogger(__name__)

# 健康探测超时（秒）：探测失败降级离线标记，不阻断管理流程
_HEALTH_TIMEOUT = 5.0

# 命名空间名约束：外部工具运行时命名为 <namespace>_<tool>，须是合法工具名片段
_NAMESPACE_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")

# 拒绝命名空间时回显的可用工具数上限：工具清单可能上百条，全量拼进错误信息过长
_HINT_TOOL_LIMIT = 20


async def _invalidate_reasoning_graphs() -> None:
    """提交成功后清空推理图缓存并广播。

    外部 MCP 工具在图构建时装载，缓存不失效则已建图仍持旧工具集（禁用 Server、
    调整命名空间后依旧可调用）；图缓存是进程内的，多实例各自失效自己的。
    """
    from app.infrastructure.cache.cache import publish_graph_invalidation
    from app.service.ai.service.reasoning_service import reasoning_service

    reasoning_service.invalidate_graph_cache()
    await publish_graph_invalidation()


class AiMcpServerService:
    # ── Server 注册表 CRUD ─────────────────────────────────

    async def create_server(self, db, form: McpServerCreate) -> McpServerResult:
        if await ai_mcp_server_repository.get_by_name(db, form.name):
            raise BusinessException(ResultCode.DATA_EXISTS, "MCP Server 名称已存在")
        if not form.endpoint:
            raise BusinessException(ResultCode.PARAM_ERROR, "端点 URL 不能为空")
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
        if data.get("name"):
            existing = await ai_mcp_server_repository.get_by_name(db, data["name"])
            if existing and existing.id != server_id:
                raise BusinessException(ResultCode.DATA_EXISTS, "MCP Server 名称已存在")
        if "endpoint" in data and not data["endpoint"]:
            raise BusinessException(ResultCode.PARAM_ERROR, "端点 URL 不能为空")
        await ai_mcp_server_repository.update(db, server, data)
        defer_after_commit(db, _invalidate_reasoning_graphs)
        return McpServerResult.model_validate(server)

    async def delete_server(self, db, server_id: int) -> None:
        server = await self._get_server_or_raise(db, server_id)
        refs = await self._count_agent_references(db, server.id)
        if refs > 0:
            raise BusinessException(
                ResultCode.DATA_BIND_EXISTS,
                f"MCP Server [{server.name}] 已被 {refs} 个 Agent 关联（命名空间），请先解绑再删除",
            )
        await ai_mcp_server_repository.soft_delete_by_ids(db, [server.id])
        defer_after_commit(db, _invalidate_reasoning_graphs)

    async def _count_agent_references(self, db, server_id: int) -> int:
        """统计关联了该 Server 命名空间的 Agent 数（同名命名空间跨 Server 复用时保守计入）。"""
        stmt = (
            select(func.count(func.distinct(SysAiAgentMcp.agent_id)))
            .join(
                SysAiMcpNamespace,
                SysAiMcpNamespace.namespace == SysAiAgentMcp.mcp_namespace,
            )
            .where(SysAiMcpNamespace.server_id == server_id)
        )
        return (await db.execute(stmt)).scalar() or 0

    async def switch_server_status(self, db, server_id: int, status: int) -> McpServerResult:
        server = await self._get_server_or_raise(db, server_id)
        server.status = status
        await db.flush()
        await db.refresh(server)
        defer_after_commit(db, _invalidate_reasoning_graphs)
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
        """合并式更新凭据：未传字段保留原值（api_key 传空不覆盖旧 Key，与前端语义一致）。

        clear=true 为凭据轮换/吊销场景，整体清空（credential_configured 归 false）。
        """
        server = await self._get_server_or_raise(db, server_id)
        if form.clear:
            server.credentials = None
        else:
            creds = dict(server.credentials or {})
            if form.api_key:
                creds["api_key"] = encrypt(form.api_key)
            if form.extra:
                extra = dict(creds.get("extra") or {})
                extra.update({k: encrypt(v) for k, v in form.extra.items()})
                creds["extra"] = extra
            server.credentials = creds
        await db.flush()
        # 装载期的工具闭包持有 Server 实体并实时取凭据，轮换后旧图仍用旧 Key 调用
        defer_after_commit(db, _invalidate_reasoning_graphs)

    # ── 健康探测（失败降级离线标记，不阻断） ────────────────────

    async def probe_health(self, db, server_id: int) -> McpHealthResult:
        server = await self._get_server_or_raise(db, server_id)
        status, latency_ms = await self._probe(server)
        server.health = status
        server.last_check_time = datetime.now()
        await db.flush()
        return McpHealthResult(status=status, latency_ms=latency_ms)

    async def _probe(self, server: SysAiMcpServer) -> tuple[str, int | None]:
        """探测外部端点连通性，返回 (status, latency_ms)；异常一律离线降级。

        streamable-http 按 MCP 协议以 POST initialize（JSON-RPC 2.0）探测，
        其余协议回退 HTTP GET；端点经统一 SSRF 守卫（预设白名单放行）。探测携带
        Server 凭据鉴权头——需鉴权的 Server 裸探测恒被 401 判离线。
        """
        endpoint = server.endpoint
        if endpoint is None or not await check_endpoint_safe(endpoint):
            return "offline", None
        auth_headers = build_mcp_auth_headers(server)
        try:
            # 禁止跟随重定向：30x 跳转目标未经 SSRF 校验，跟随即绕过端点防护
            async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT, follow_redirects=False) as client:
                start = time.monotonic()
                if server.protocol_type == "streamable-http":
                    resp = await client.post(
                        endpoint,
                        headers={**auth_headers, "Content-Type": "application/json"},
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
                    resp = await client.get(endpoint, headers=auth_headers)
                latency_ms = int((time.monotonic() - start) * 1000)
            # 仅 2xx 判在线；重定向不跟随（SSRF），3xx/4xx/5xx 均离线
            return ("online" if resp.status_code < 300 else "offline"), latency_ms
        except (httpx.HTTPError, Exception) as exc:
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
        known = await self._known_tool_names(db, server_id)
        seen: set[str] = set()
        for item in items:
            if not _NAMESPACE_NAME_RE.match(item.name or ""):
                raise BusinessException(
                    ResultCode.PARAM_ERROR,
                    f"命名空间名非法（须字母开头，仅含字母/数字/下划线/连字符，≤64字符）: "
                    f"{item.name}",
                )
            if item.name in seen:
                raise BusinessException(ResultCode.PARAM_ERROR, f"命名空间重复: {item.name}")
            seen.add(item.name)
            if any(not t or len(t) > 256 for t in item.toolNames):
                raise BusinessException(
                    ResultCode.PARAM_ERROR, f"命名空间 {item.name} 的工具名不能为空且≤256字符"
                )
            self._reject_unknown_tools(item, known)
        await ai_mcp_namespace_repository.delete_by_server(db, server_id)
        for item in items:
            await ai_mcp_namespace_repository.create(
                db,
                SysAiMcpNamespace(
                    server_id=server_id, namespace=item.name, tool_names=item.toolNames
                ),
            )
        await db.flush()
        defer_after_commit(db, _invalidate_reasoning_graphs)
        return await self.list_namespaces(db, server_id)

    @staticmethod
    async def _known_tool_names(db, server_id: int) -> set[str]:
        return {t.name for t in await ai_mcp_tool_repository.list_by_server(db, server_id)}

    @staticmethod
    def _reject_unknown_tools(item: McpNamespaceItem, known: set[str]) -> None:
        """拒绝清单外的工具名：拼错的工具名装载期静默失效，且无提示可自查。"""
        unknown = [t for t in item.toolNames if t not in known]
        if not unknown:
            return
        hint = "、".join(sorted(known)[:_HINT_TOOL_LIMIT]) or "无（请先在工具 Tab 拉取清单）"
        raise BusinessException(
            ResultCode.PARAM_ERROR,
            f"命名空间 {item.name} 含未拉取到的工具: {', '.join(unknown)}；当前可用工具: {hint}",
        )

    # ── 工具 ──────────────────────────────────────────────

    async def _get_server_or_raise(self, db, server_id: int) -> SysAiMcpServer:
        server = await ai_mcp_server_repository.get_by_id(db, server_id)
        if not server:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "MCP Server 不存在")
        return server


ai_mcp_server_service = AiMcpServerService()
