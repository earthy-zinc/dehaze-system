"""外部 MCP Server 工具运行时装载器

将 Agent 关联命名空间对应的外部 MCP Server 工具装载为 LangChain StructuredTool，
使 LLM 经 Function Calling 原生调用外部 Server（对齐 MCP 集成规范：tools/list →
tools/call 直接映射为工具），而非仅靠文本注入。

实现要点：
- 按 agent.mcp_namespaces 反向关联到启用的外部 Server（sys_ai_mcp_namespace）
- 装载时连接 Server 拉取工具清单（带凭据鉴权头 + SSRF 前置校验），超时降级为空，
  不阻塞构图
- 工具命名 <namespace>_<tool>（与 Guardrail 命名空间越权校验约定一致）
- 工具执行时建立短会话调用 tools/call（streamable-http 无状态），记录调用审计
  （sys_ai_mcp_call，与调用审计面板打通）
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import AsyncExitStack
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from app.database import get_db_session
from app.models.entity.sys_ai_mcp_server import SysAiMcpServer
from app.repository.ai_mcp_namespace_repository import ai_mcp_namespace_repository
from app.repository.ai_mcp_server_repository import ai_mcp_server_repository
from app.service.ai_mcp.mcp_connection import (
    apply_ssrf_guard,
    build_mcp_auth_headers,
    call_remote_tool,
)
from app.service.ai_mcp.mcp_manage_service import mcp_manage_service

logger = logging.getLogger(__name__)

# 装载单 Server 超时（秒）：外部 Server 抖动降级为空，不阻塞构图
_LOAD_TIMEOUT = 10.0
# 单次工具调用超时（秒）
_CALL_TIMEOUT = 60.0


def _schema_py_type(schema: dict[str, Any]) -> Any:
    """将 JSON Schema 基础类型映射为 pydantic 类型（嵌套 array/object 简化处理）。"""
    stype = schema.get("type")
    enum = schema.get("enum")
    if enum:
        from typing import Literal

        return Literal[tuple(enum)]  # type: ignore[valid-type]
    if stype == "array":

        return list[_schema_py_type(schema.get("items") or {"type": "string"})]  # type: ignore[index]
    if stype == "object":
        return dict
    if stype == "integer":
        return int
    if stype == "number":
        return float
    if stype == "boolean":
        return bool
    return str


def _build_args_schema(input_schema: dict[str, Any]) -> type[BaseModel] | None:
    """从 MCP inputSchema 构造 pydantic 参数模型（供 LLM 拿到完整参数契约）。

    无 properties（无参工具）返回 None，工具以无参函数形式注册。
    """
    properties = (input_schema or {}).get("properties") or {}
    if not properties:
        return None
    required = set((input_schema or {}).get("required") or [])
    fields: dict[str, tuple[Any, Any]] = {}
    for name, prop in properties.items():
        py_type = _schema_py_type(prop or {})
        desc = (prop or {}).get("description") or ""
        if name in required:
            fields[name] = (py_type, Field(..., description=desc))
        else:
            fields[name] = (
                py_type | None,
                Field(None, description=desc),
            )
    return create_model("McpToolArgs", **fields)  # type: ignore[call-overload]


class McpExternalToolLoader:
    """外部 MCP Server 工具运行时装载器。"""

    async def load_tools(
        self,
        db,
        agent_namespaces: list[str],
        ctx: dict[str, Any],
    ) -> list[StructuredTool]:
        """装载 Agent 命名空间对应的外部 Server 工具（失败降级为空）。"""
        if not agent_namespaces:
            return []
        wanted = set(agent_namespaces)
        servers = await ai_mcp_server_repository.list_enabled(db)
        tools: list[StructuredTool] = []
        for server in servers:
            try:
                tools.extend(
                    await asyncio.wait_for(
                        self._load_server_tools(db, server, wanted, ctx),
                        timeout=_LOAD_TIMEOUT,
                    )
                )
            except TimeoutError:
                logger.warning("MCP 外部工具装载超时: server=%s", server.name)
            except Exception as exc:  # noqa: BLE001 单 Server 失败不影响整体构图
                logger.warning("MCP 外部工具装载失败: server=%s err=%s", server.name, exc)
        if tools:
            logger.info("MCP 外部工具装载完成，共 %s 个", len(tools))
        return tools

    async def _load_server_tools(
        self,
        db,
        server: SysAiMcpServer,
        wanted: set[str],
        ctx: dict[str, Any],
    ) -> list[StructuredTool]:
        """装载单个 Server 中属于命中命名空间的工具。

        命名空间配置了 tool_names（归属工具清单）时按清单过滤；未配置时该
        命名空间承载 Server 全部工具。
        """
        if server.protocol_type == "stdio" or not server.endpoint:
            return []
        allowed, _reason = await apply_ssrf_guard(server)
        if not allowed:
            return []
        rows = await ai_mcp_namespace_repository.list_by_server(db, server.id)
        # ns -> tool_names 归属清单（空表示承载全部工具）
        ns_scope: dict[str, set[str]] = {}
        for row in rows:
            if row.namespace in wanted:
                names = set(row.tool_names or [])
                ns_scope[row.namespace] = names
        if not ns_scope:
            return []
        listed = await self._list_tools(server)
        if not listed:
            return []
        result: list[StructuredTool] = []
        for ns, scope in ns_scope.items():
            for t in listed:
                if scope and t["name"] not in scope:
                    continue
                result.append(self._build_tool(server, ns, t, ctx))
        return result

    async def _list_tools(self, server: SysAiMcpServer) -> list[dict[str, Any]]:
        """连接 Server 拉取工具清单（带鉴权头），失败返回空。"""
        headers = build_mcp_auth_headers(server)
        try:
            from mcp.client.session import ClientSession
            from mcp.client.sse import sse_client
            from mcp.client.streamable_http import streamablehttp_client

            client_fn = (
                sse_client if server.protocol_type == "sse" else streamablehttp_client
            )
            async with AsyncExitStack() as stack:
                transport = client_fn(server.endpoint, headers=headers)
                read, write, _ = await stack.enter_async_context(transport)
                session = await stack.enter_async_context(ClientSession(read, write))
                listed = await asyncio.wait_for(session.list_tools(), _LOAD_TIMEOUT)
                return [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "input_schema": dict(t.inputSchema or {}),
                    }
                    for t in listed.tools or []
                ]
        except Exception as exc:  # noqa: BLE001 外部 Server 不可用降级为空
            logger.warning("MCP 外部 Server 工具清单拉取失败: server=%s err=%s", server.name, exc)
            return []

    def _build_tool(
        self,
        server: SysAiMcpServer,
        namespace: str,
        tool: dict[str, Any],
        ctx: dict[str, Any],
    ) -> StructuredTool:
        """构造可调用的 StructuredTool（<namespace>_<tool> 命名 + 调用审计）。"""
        raw_name = tool["name"]
        name = f"{namespace}_{raw_name}"
        desc = tool.get("description") or f"调用外部 MCP Server「{server.name}」的工具 {raw_name}"
        args_schema = _build_args_schema(tool.get("input_schema") or {})
        server_id = server.id
        server_name = server.name

        async def _call(**kwargs: Any) -> str:
            started = time.monotonic()
            request_text = json.dumps(kwargs, ensure_ascii=False)[:2000]
            try:
                text = await call_remote_tool(server, raw_name, kwargs)
                result = "success"
            except Exception as exc:  # noqa: BLE001 错误作为工具结果回喂模型
                logger.warning(
                    "MCP 外部工具调用失败: server=%s tool=%s err=%s",
                    server_name,
                    raw_name,
                    exc,
                )
                text = f"工具调用失败: {exc}"
                result = "failure"
            latency_ms = int((time.monotonic() - started) * 1000)
            await self._record_call(
                ctx,
                server_id,
                server_name,
                raw_name,
                result,
                latency_ms,
                request_text,
                text[:2000],
            )
            return text

        # 显式传 coroutine：from_function 对闭包 async 函数存在误判（当作 sync），
        # 导致 ainvoke 返回未 await 的 coroutine；显式指定后走 coroutine 执行路径。
        return StructuredTool.from_function(
            func=_call,
            coroutine=_call,
            name=name,
            description=desc,
            args_schema=args_schema,
        )

    async def _record_call(
        self,
        ctx: dict[str, Any],
        server_id: int,
        server_name: str,
        tool_name: str,
        result: str,
        latency_ms: int,
        request: str,
        response: str,
    ) -> None:
        """落库调用审计（失败不阻断工具调用）。"""
        try:
            async with get_db_session() as db:
                await mcp_manage_service.record_call(
                    db,
                    user_id=ctx.get("user_id"),
                    server_id=server_id,
                    server_name=server_name,
                    tool_name=tool_name,
                    result=result,
                    latency_ms=latency_ms,
                    request={"arguments": request},
                    response=response,
                )
        except Exception as exc:  # noqa: BLE001 审计失败不影响工具调用
            logger.warning(
                "MCP 外部调用审计写入失败: server=%s tool=%s err=%s",
                server_id,
                tool_name,
                exc,
            )


mcp_external_tool_loader = McpExternalToolLoader()
