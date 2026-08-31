"""外部 MCP Server 连接公共工具

- build_mcp_auth_headers：从 Server 凭据（AES 密文）构造鉴权头，供工具拉取与
  运行时调用统一使用（api_key / oauth2 access_token 均以 Bearer 形式携带）。
- check_endpoint_safe：端点安全校验——市场内置预设（本平台受信内网自建
  Server）放行，其余须通过 SSRF 校验（https + 禁内网 + 防 DNS 重绑定）。
- apply_ssrf_guard：连接前置守卫，返回 (是否允许, 拒绝原因)。
- call_remote_tool：调用外部 Server 的 tools/call（带鉴权头 + 超时），供运行时
  工具执行与管理员试调用复用同一连接路径。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.infrastructure.crypto.aes_cipher import decrypt
from app.models.entity.sys_ai_mcp_server import SysAiMcpServer
from app.service.ai_mcp.mcp_presets import PRESET_ENDPOINTS
from app.utils.ssrf import is_safe_url

logger = logging.getLogger(__name__)


def _decrypt_field(value: Any) -> str | None:
    """解密单字段密文，解密失败返回 None（不阻断调用，仅缺鉴权）。"""
    if not isinstance(value, str) or not value:
        return None
    try:
        return decrypt(value)
    except Exception:  # noqa: BLE001 密文损坏降级为无鉴权，不抛
        logger.warning("MCP 凭据解密失败，降级为无鉴权调用")
        return None


def build_mcp_auth_headers(server: SysAiMcpServer | None) -> dict[str, str]:
    """从 Server 凭据构造鉴权头。

    - api_key 密文 → Authorization: Bearer <key>
    - oauth2 access_token（extra.access_token 密文）→ Authorization: Bearer <token>
    - 其余（oauth2 未配置 token / none）→ 空（无鉴权）
    """
    if server is None or not server.credentials:
        return {}
    creds = server.credentials or {}
    api_key = _decrypt_field(creds.get("api_key"))
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    extra = creds.get("extra")
    if isinstance(extra, dict):
        token = _decrypt_field(extra.get("access_token"))
        if token:
            return {"Authorization": f"Bearer {token}"}
    return {}


async def check_endpoint_safe(endpoint: str | None) -> bool:
    """端点是否允许连接：市场内置预设放行，其余须通过完整 SSRF 校验。

    预设（本平台自建本地 Server）走白名单；自定义 Server 必须是 https 且非内网，
    防止拉取/调用路径经代理访问内网（与健康探测的 SSRF 防护保持一致）。
    """
    if not endpoint:
        return False
    if endpoint in PRESET_ENDPOINTS:
        return True
    return await is_safe_url(endpoint)


# 工具调用单次超时（秒）
_CALL_TIMEOUT = 60.0


async def call_remote_tool(
    server: SysAiMcpServer,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """调用外部 Server 的 tools/call，返回结果文本。

    经 URL 型协议（streamable-http/sse）建立短会话（带凭据鉴权头）调用；
    供运行时工具执行与管理员试调用复用，超时抛异常由调用方处理。
    """
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamablehttp_client

    headers = build_mcp_auth_headers(server)
    client_fn = (
        sse_client if server.protocol_type == "sse" else streamablehttp_client
    )
    async with asyncio.timeout(_CALL_TIMEOUT):
        async with client_fn(server.endpoint, headers=headers) as (read, write, _):
            session = ClientSession(read, write)
            async with session:
                result = await session.call_tool(tool_name, arguments)
    return _extract_tool_text(result)


def _extract_tool_text(result: Any) -> str:
    """从 MCP CallToolResult.content 提取文本（首个 text 块拼接）。"""
    content = getattr(result, "content", None) or []
    texts = [
        block.text
        for block in content
        if getattr(block, "text", None) is not None
    ]
    if texts:
        return "\n".join(texts)
    return str(content) if content else "(无返回内容)"


async def apply_ssrf_guard(server: SysAiMcpServer | None) -> tuple[bool, str]:
    """连接前置守卫：端点不合法时返回 (False, 原因)，合法返回 (True, "")。

    供工具拉取/运行时装载统一调用，避免各调用方重复校验逻辑。
    """
    if server is None:
        return False, "MCP Server 不存在"
    if server.protocol_type == "stdio":
        return True, ""  # 本地进程，无网络面
    if not server.endpoint:
        return False, "MCP Server 未配置端点"
    if not await check_endpoint_safe(server.endpoint):
        return False, "MCP 端点不安全（仅允许 https 且禁止内网地址）"
    return True, ""
