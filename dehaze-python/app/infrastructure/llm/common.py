"""LLM 协议层共享类型与工具

被 model_client（抽象接口 + 工厂）与具体协议客户端（anthropic / openai_compat）
共同引用，不依赖任何具体实现，故无环。协议类型取值与 sys_ai_provider.protocol_type 对齐。
"""

from dataclasses import dataclass

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
