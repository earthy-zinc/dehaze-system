"""LLM 协议客户端统一接口与工厂

按 sys_ai_provider.protocol_type 分发到具体协议实现（openai_compat / anthropic）。
LlmClient 只依赖此接口做编排；LlmStreamChunk / build_auth_headers 为协议层
共享类型与工具，供协议实现与连通性测试复用。
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Protocol

import httpx

# 协议类型（与 sys_ai_provider.protocol_type 取值对齐）
PROTOCOL_OPENAI_COMPAT = "openai_compat"
PROTOCOL_ANTHROPIC = "anthropic"

# 调用失败错误码
_ERROR_5XX = "5xx"


@dataclass
class LlmStreamChunk:
    """统一的 LLM 流式响应块"""

    # type: text_delta / thinking_delta / tool_call_start / tool_call_delta / tool_call_complete / done
    type: str
    content: str = ""
    usage: dict | None = None
    tool_call_id: str = ""
    tool_call_name: str = ""


def _map_httpx_error(exc: Exception) -> tuple[str, str]:
    """将 httpx 异常映射为 (error_code, detail)"""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if 500 <= status <= 599:
            return _ERROR_5XX, f"供应商服务端错误: HTTP {status}"
        return str(status), f"供应商返回 HTTP {status}"
    if isinstance(exc, httpx.ConnectError):
        return "connection", "供应商连接失败"
    if isinstance(exc, httpx.TimeoutException):
        return "timeout", "供应商请求超时"
    if isinstance(exc, httpx.TransportError):
        return "transport", "供应商传输错误"
    return "unknown", str(exc)


def build_auth_headers(provider, api_key: str) -> dict:
    """按供应商认证方式构建请求头（合并 default_headers）。

    LlmClient 与连通性测试共用，认证头组装单一实现。
    """
    headers = dict(provider.default_headers or {})
    if provider.auth_type == "bearer":
        headers["Authorization"] = f"Bearer {api_key}"
    elif provider.auth_type == "x-api-key":
        headers["x-api-key"] = api_key
    elif provider.auth_type == "custom":
        # 自定义认证头：头名在 default_headers 中以 auth_header 键配置
        header_name = headers.pop("auth_header", "Authorization")
        headers[header_name] = api_key
    return headers


class ChatModelClient(Protocol):
    """LLM 流式对话客户端统一接口（按协议实现，注入共享 httpx 连接）"""

    def stream_chat(
        self,
        provider,
        api_key: str,
        model,
        messages: list[dict],
        system_prompt: str | None,
        max_tokens: int | None,
        tools: list[dict] | None,
        tool_choice: str | None,
        temperature: float = 0.7,
    ) -> AsyncGenerator[LlmStreamChunk, None]:
        """构建协议原生请求并解析 SSE，产出统一 LlmStreamChunk 流"""


def create_chat_client(protocol_type: str, client: httpx.AsyncClient) -> ChatModelClient:
    """按协议类型创建流式对话客户端（未知协议回退 OpenAI 兼容）"""
    if protocol_type == PROTOCOL_ANTHROPIC:
        return AnthropicClient(client)
    return OpenAiCompatClient(client)


from app.infrastructure.llm.anthropic_client import AnthropicClient
from app.infrastructure.llm.openai_compat_client import OpenAiCompatClient
