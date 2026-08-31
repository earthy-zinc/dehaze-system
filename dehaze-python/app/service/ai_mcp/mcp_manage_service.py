"""外部 MCP Server 工具/市场/调用审计管理服务（F-M08-006 §2.6.13）。

承担外部 MCP Server 接入后的工具拉取、市场目录与一键接入、调用审计职责。
Server 注册表（注册/启停/健康/凭据）与命名空间配置由 mcp-core 的
AiMcpServerService 承载，本服务聚焦工具消费侧与审计侧。

工具拉取：从远端按 MCP 协议（streamable-http/sse）拉取工具清单写入
sys_ai_mcp_tool（覆盖式重建），远端不可达时落空工具、不阻断管理流程。
"""

import logging
import time
from typing import Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_mcp_call import SysAiMcpCall
from app.models.entity.sys_ai_mcp_server import SysAiMcpServer
from app.models.entity.sys_ai_mcp_tool import SysAiMcpTool
from app.models.schema.ai_mcp import (
    McpCallQuery,
    McpCallResult,
    McpMarketPreset,
    McpServerResult,
    McpToolResult,
    McpToolTestResult,
)
from app.models.schema.common import PageResult
from app.repository.ai_mcp_call_repository import ai_mcp_call_repository
from app.repository.ai_mcp_server_repository import ai_mcp_server_repository
from app.repository.ai_mcp_tool_repository import ai_mcp_tool_repository
from app.service.ai_mcp.mcp_connection import apply_ssrf_guard, call_remote_tool
from app.service.ai_mcp.mcp_presets import MARKET_PRESETS as _MARKET_PRESETS
from app.service.ai_mcp.mcp_tool_fetcher import McpToolFetcher, mcp_tool_fetcher

logger = logging.getLogger(__name__)


class McpManageService:
    """外部 MCP Server 工具/市场/调用审计管理。"""

    def __init__(self, fetcher: McpToolFetcher | None = None) -> None:
        self.fetcher = fetcher if fetcher is not None else mcp_tool_fetcher

    # ── 工具 ──────────────────────────────────────────────

    async def get_tools(self, db: AsyncSession, server_id: int) -> list[McpToolResult]:
        """拉取 Server 工具清单并覆盖式写入 sys_ai_mcp_tool，返回工具详情列表。

        从远端按协议拉取 tools（name/description/input_schema），落库前清空该
        Server 旧工具（工具随 Server 拉取重建，不保留历史版本）。远端不可达
        时落空工具，并更新 server.tool_count 冗余值。
        """
        server = await self._get_server_or_404(db, server_id)
        try:
            fetched = await self.fetcher.list_tools(server)
        except Exception as exc:  # noqa: BLE001 拉取失败不阻断管理，落空工具
            logger.warning("MCP 工具拉取异常: server_id=%s err=%s", server.id, exc)
            fetched = []
        await ai_mcp_tool_repository.delete_by_server(db, server.id)
        tools = [
            SysAiMcpTool(
                server_id=server.id,
                name=t.get("name", ""),
                description=t.get("description"),
                input_schema=t.get("input_schema"),
            )
            for t in fetched
            if t.get("name")
        ]
        if tools:
            await ai_mcp_tool_repository.create_all(db, tools)
        await ai_mcp_server_repository.update(db, server, {"tool_count": len(tools)})
        rows = await ai_mcp_tool_repository.list_by_server(db, server.id)
        return [self._to_tool(t) for t in rows]

    # ── 工具试调用 ──────────────────────────────────────────

    async def test_tool(
        self,
        db: AsyncSession,
        server_id: int,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> McpToolTestResult:
        """试调用外部 MCP Server 工具（管理员验证连通性/参数，不走 LLM）。

        连接复用 call_remote_tool（凭据鉴权头 + SSRF 守卫），结果与失败均落
        调用审计；失败不抛异常，结果结构透出错误信息供前端展示。
        """
        server = await self._get_server_or_404(db, server_id)
        allowed, reason = await apply_ssrf_guard(server)
        if not allowed:
            raise BusinessException(ResultCode.PARAM_ERROR, reason)
        started = time.monotonic()
        try:
            text = await call_remote_tool(server, tool_name, arguments or {})
            result = "success"
            error = None
        except Exception as exc:  # noqa: BLE001 试调用失败透出错误，不抛
            logger.warning(
                "MCP 工具试调用失败: server=%s tool=%s err=%s",
                server.name,
                tool_name,
                exc,
            )
            text = f"工具调用失败: {exc}"
            result = "failure"
            error = str(exc)
        latency_ms = int((time.monotonic() - started) * 1000)
        await self.record_call(
            db,
            user_id=None,
            server_id=server.id,
            server_name=server.name,
            tool_name=tool_name,
            result=result,
            latency_ms=latency_ms,
            request={"arguments": arguments or {}},
            response=text[:2000],
        )
        await db.flush()
        return McpToolTestResult(
            success=(result == "success"),
            result=text[:8000] if result == "success" else None,
            error=error,
            latency_ms=latency_ms,
        )

    # ── 市场 ──────────────────────────────────────────────

    async def get_market(self, db: AsyncSession) -> list[McpMarketPreset]:
        """返回市场预设目录，installed 由是否存在同名 Server 推导。"""
        items = []
        for preset in _MARKET_PRESETS:
            server = await ai_mcp_server_repository.get_by_name(
                db, preset["name"], include_deleted=True
            )
            items.append(
                McpMarketPreset(
                    preset_id=preset["preset_id"],
                    name=preset["name"],
                    description=preset["description"],
                    capability_tags=preset["capability_tags"],
                    installed=server is not None,
                )
            )
        return items

    async def install_preset(
        self, db: AsyncSession, redis: Redis, preset_id: str
    ) -> McpServerResult:
        """从市场一键接入预设：注册 Server（重名复用）并拉取工具清单。"""
        preset = next((p for p in _MARKET_PRESETS if p["preset_id"] == preset_id), None)
        if not preset:
            raise BusinessException(ResultCode.PARAM_ERROR, "未知的市场预设")

        server = await ai_mcp_server_repository.get_by_name(
            db, preset["name"], include_deleted=True
        )
        if server is None:
            server = SysAiMcpServer(
                name=preset["name"],
                description=preset["description"],
                protocol_type=preset["protocol_type"],
                endpoint=preset["endpoint"],
                auth_type=preset["auth_type"],
                status=1,
                tool_count=0,
            )
            await ai_mcp_server_repository.create(db, server)
        tools = await self.get_tools(db, server.id)
        server.tool_count = len(tools)
        await db.flush()
        return self._to_server(server)

    # ── 调用审计 ──────────────────────────────────────────

    async def record_call(
        self,
        db: AsyncSession,
        *,
        user_id: int | None,
        server_id: int,
        server_name: str | None,
        tool_name: str,
        result: str,
        latency_ms: int | None = None,
        request: dict[str, Any] | None = None,
        response: str | None = None,
    ) -> None:
        """记录外部 MCP 工具调用审计（只追加，成功与失败均落一条）。

        审计为对账与安全基线，写入失败不阻断工具调用主流程。
        """
        try:
            await ai_mcp_call_repository.create(
                db,
                SysAiMcpCall(
                    user_id=user_id,
                    server_id=server_id,
                    server_name=server_name,
                    tool_name=tool_name,
                    result=result,
                    status=1 if result == "success" else 0,
                    latency_ms=latency_ms,
                    request=request,
                    response=response,
                ),
            )
        except Exception as exc:  # noqa: BLE001 审计失败不阻断调用
            logger.warning(
                "MCP 调用审计写入失败: server_id=%s tool=%s err=%s",
                server_id,
                tool_name,
                exc,
            )

    async def list_calls(
        self, db: AsyncSession, query: McpCallQuery
    ) -> PageResult[McpCallResult]:
        """分页查询外部 MCP 工具调用审计（create_time 倒序）。"""
        rows, total = await ai_mcp_call_repository.paginate_calls(
            db,
            query.pageNum,
            query.pageSize,
            server_id=query.server_id,
            tool_name=query.tool_name,
        )
        return PageResult(list=[self._to_call(c) for c in rows], total=total)

    # ── 装配 ──────────────────────────────────────────────

    async def _get_server_or_404(self, db: AsyncSession, server_id: int) -> SysAiMcpServer:
        server = await ai_mcp_server_repository.get_by_id(db, server_id)
        if not server:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "MCP Server 不存在")
        return server

    @staticmethod
    def _to_tool(tool: SysAiMcpTool) -> McpToolResult:
        return McpToolResult(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
        )

    @staticmethod
    def _to_call(call: SysAiMcpCall) -> McpCallResult:
        return McpCallResult(
            id=call.id,
            user_id=call.user_id,
            server_id=call.server_id,
            server_name=call.server_name,
            tool_name=call.tool_name,
            result=call.result,
            latency_ms=call.latency_ms,
            create_time=call.create_time,
        )

    @staticmethod
    def _to_server(server: SysAiMcpServer) -> McpServerResult:
        return McpServerResult(
            id=server.id,
            name=server.name,
            description=server.description,
            protocol_type=server.protocol_type,
            endpoint=server.endpoint,
            auth_type=server.auth_type,
            status=server.status,
            health=server.health,
            tool_count=server.tool_count,
            create_time=server.create_time,
            update_time=server.update_time,
        )


mcp_manage_service = McpManageService()
