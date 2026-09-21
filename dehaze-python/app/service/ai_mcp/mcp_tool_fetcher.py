"""外部 MCP Server 工具拉取客户端

按 server 的传输协议（streamable-http / sse）连接远端 MCP Server，调用
`tools/list` 拉取工具清单（name + description + inputSchema）。仅支持 URL 型
协议——本地进程型（stdio）无网络端点，注册入口已不接受。

连接失败抛异常，由调用方决定降级口径（管理侧刷新落空工具、市场接入侧置失败态）。

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
        """按 server 协议拉取工具列表，连接失败抛异常。

        返回 [{"name", "description", "input_schema"}]，与 sys_ai_mcp_tool
        落库字段对齐；无端点或端点被 SSRF 守卫拦截时返回空。
        """
        if not server.endpoint:
            return []
        # SSRF 前置守卫：端点不安全（非预设 + 非 https/内网）直接拒绝，不发起连接
        allowed, reason = await apply_ssrf_guard(server)
        if not allowed:
            logger.warning(
                "MCP 工具拉取被 SSRF 守卫拦截: server_id=%s reason=%s", server.id, reason
            )
            return []
        return await self._list_by_url(server)

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
        # 传输拆除失败不能覆盖 list_tools 的结果/异常（上层据此决定降级口径），
        # 但必须可见，否则连接泄漏与协议关闭错误无从排查。
        if self._client:
            try:
                await self._client.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.warning("关闭 MCP 传输失败: %s", e, exc_info=True)

    def session(self) -> ClientSession:
        return ClientSession(self._read, self._write)


mcp_tool_fetcher = McpToolFetcher()
