"""外部 MCP Server 工具拉取客户端

按 server 的传输协议（streamable-http / sse）连接远端 MCP Server，调用
`tools/list` 拉取工具清单（name + description + inputSchema）。

stdio 协议需要本地进程命令，而注册表未配置 command 字段，故仅支持 URL 型
协议（streamable-http / sse）；stdio 的 server 拉取返回空列表，由调用方按
空工具处理。

服务层经构造注入使用本类（默认单例），测试注入 mock fetcher 避免真实联网。
"""

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

from app.service.ai_mcp.mcp_connection import apply_ssrf_guard, build_mcp_auth_headers

logger = logging.getLogger(__name__)

# 拉取单次超时（秒）：外部 Server 网络抖动时降级为空，不阻塞管理流程
_FETCH_TIMEOUT = 10.0


class McpToolFetcher:
    """从外部 MCP Server 拉取工具清单。"""

    async def list_tools(self, server) -> list[dict[str, Any]]:
        """按 server 协议拉取工具列表，失败返回空列表。

        返回 [{"name", "description", "input_schema"}]，与 sys_ai_mcp_tool
        落库字段对齐；协议不支持（stdio）或无端点时返回空。
        """
        if server.protocol_type == "stdio" or not server.endpoint:
            return []
        # SSRF 前置守卫：端点不安全（非预设 + 非 https/内网）直接拒绝，不发起连接
        allowed, reason = await apply_ssrf_guard(server)
        if not allowed:
            logger.warning(
                "MCP 工具拉取被 SSRF 守卫拦截: server_id=%s reason=%s", server.id, reason
            )
            return []
        try:
            return await self._list_by_url(server)
        except Exception as exc:  # noqa: BLE001 拉取失败不阻断管理，落空工具
            logger.warning("MCP 工具拉取失败: server_id=%s err=%s", server.id, exc)
            return []

    async def _list_by_url(self, server) -> list[dict[str, Any]]:
        """经 URL 型协议（streamable-http/sse）连接并拉取工具。

        连接时注入 Server 凭据鉴权头（api_key/oauth2 token，AES 解密）。
        """
        headers = build_mcp_auth_headers(server)
        client_fn = sse_client if server.protocol_type == "sse" else streamablehttp_client
        async with AsyncExitStack() as stack:
            transport = _McpTransport(client_fn, server.endpoint, headers)
            await stack.enter_async_context(transport)
            session = await stack.enter_async_context(transport.session())
            result = await asyncio.wait_for(session.list_tools(), _FETCH_TIMEOUT)
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": dict(t.inputSchema or {}),
                }
                for t in result.tools or []
            ]


class _McpTransport:
    """MCP 底层传输上下文：统一封装底层 client 的读写流，供 ClientSession 使用。"""

    def __init__(self, client_fn, url: str, headers: dict[str, str] | None = None):
        self._client_fn = client_fn
        self._url = url
        self._headers = headers or {}
        self._client = None

    async def __aenter__(self):
        self._client = await self._client_fn(self._url, headers=self._headers)
        self._read, self._write, _ = await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            try:
                await self._client.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:  # noqa: BLE001 关闭异常不影响调用方
                pass

    def session(self) -> ClientSession:
        return ClientSession(self._read, self._write)


mcp_tool_fetcher = McpToolFetcher()
