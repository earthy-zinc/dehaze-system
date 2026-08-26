"""MCP 网关客户端

连接 MCP 能力网关（dehaze-mcp-gateway），通过 JSON-RPC 2.0 调用 3 个元 tool：
- lookup_tool: 搜索后端 API
- lookup_tool_param_schema: 查看工具完整参数定义
- execute_tool: 调用工具

MCP 网关地址由配置 MCP_GATEWAY_URL 指定，默认 http://127.0.0.1:8082/mcp
"""

import asyncio
import json
import logging
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from app.config import settings

logger = logging.getLogger(__name__)

# list_tools 单独调用超时（秒）：预筛选为可选能力，超时即降级，不阻塞模型调用
_MCP_LIST_TIMEOUT = 5.0


class McpGatewayClient:
    """MCP 网关客户端

    通过 MCP Python SDK 的 ClientSession 连接网关，调用 3 个元 tool。
    每次调用建立新会话（MCP 网关无状态，开销可接受）。
    """

    async def lookup_tool(self, query: str) -> str:
        """搜索后端 API，返回工具名、描述和参数名列表"""
        try:
            async with self._session() as session:
                result = await session.call_tool("lookup_tool", {"query": query})
                return _extract_text(result)
        except Exception as e:
            logger.warning("MCP lookup_tool failed: %s", e)
            return ""

    async def lookup_tool_param_schema(self, tool_name: str) -> str:
        """查看指定工具的完整参数定义"""
        try:
            async with self._session() as session:
                result = await session.call_tool(
                    "lookup_tool_param_schema", {"tool_name": tool_name}
                )
                return _extract_text(result)
        except Exception as e:
            logger.warning("MCP lookup_tool_param_schema failed: %s", e)
            return ""

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """调用指定工具，执行后端 API 调用"""
        try:
            async with self._session() as session:
                result = await session.call_tool(
                    "execute_tool",
                    {"tool_name": tool_name, "arguments": json.dumps(arguments)},
                )
                return _extract_text(result)
        except Exception as e:
            logger.warning("MCP execute_tool failed: %s", e)
            return f"工具执行失败: {e}"

    async def list_tools(self) -> list[dict[str, Any]]:
        """获取网关注册的全部工具摘要（name + description + inputSchema）。

        MCP 网关基于 MCPServer，ClientSession.list_tools 返回其注册的工具列表。
        网关现状仅有 3 个元 tool，无命名空间分组；命名空间摘要由
        McpNamespacePrefilter 按工具名前缀推导。网关不可用时返回空列表。

        注意：MCP 会话基于 anyio 取消域，在 langgraph 模型调用上下文（awrap_model_call）
        中直接创建/关闭会话，请求失败时取消域可能跨任务泄漏并取消模型节点。故将整个
        会话生命周期封装进独立任务执行，隔离取消域；网关不可用一律降级为空列表。
        """
        task = asyncio.create_task(self._list_tools_impl())
        try:
            return await asyncio.wait_for(task, timeout=_MCP_LIST_TIMEOUT)
        except BaseException as e:
            task.cancel()
            logger.warning("MCP list_tools failed: %s", e)
            return []

    async def _list_tools_impl(self) -> list[dict[str, Any]]:
        try:
            async with self._session() as session:
                result = await session.list_tools()
                tools = []
                for t in result.tools or []:
                    tools.append(
                        {
                            "name": t.name,
                            "description": t.description or "",
                            "input_schema": dict(t.inputSchema or {}),
                        }
                    )
                return tools
        except BaseException as e:
            logger.warning("MCP list_tools impl failed: %s", e)
            return []

    def _session(self):
        """创建 MCP 客户端会话上下文管理器"""
        return _McpSessionContext(settings.MCP_GATEWAY_URL)


def _extract_text(result) -> str:
    """从 MCP CallToolResult 提取文本内容"""
    if not result.content:
        return ""
    # MCP 返回 content 列表，取第一个 text 块
    for block in result.content:
        if hasattr(block, "text"):
            return block.text
    return str(result.content)


class _McpSessionContext:
    """MCP 会话上下文管理器：封装 streamablehttp + ClientSession 的嵌套上下文"""

    def __init__(self, url: str):
        self._url = url
        self._http_ctx = None
        self._session_ctx = None
        self._session = None

    async def __aenter__(self) -> ClientSession:
        self._http_ctx = streamablehttp_client(self._url)
        read, write, _ = await self._http_ctx.__aenter__()
        self._session_ctx = ClientSession(read, write)
        self._session = await self._session_ctx.__aenter__()
        return self._session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session_ctx:
            try:
                await self._session_ctx.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass
        if self._http_ctx:
            try:
                await self._http_ctx.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass


mcp_gateway_client = McpGatewayClient()
