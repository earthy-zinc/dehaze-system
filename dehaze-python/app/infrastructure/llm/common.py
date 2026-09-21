"""LLM 协议层共享类型与工具

被 model_client（抽象接口 + 工厂）与具体协议客户端（anthropic / openai_compat）
共同引用，不依赖任何具体实现，故无环。协议类型取值与 sys_ai_provider.protocol_type 对齐。
"""

import hashlib
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


def build_user_identity(provider, user_id: int | None) -> tuple[str, str] | None:
    """按供应商透传配置生成用户身份标识，返回 (注入字段路径, 值)；不注入返回 None。

    值 = prefix + sha256(userId) 十六进制，按 max_len 截断（AI模型管理 §2.7.8，
    不透传任何用户隐私信息）。内置本地模型 / 未配置 / 未启用 / 无用户上下文均不注入。
    """
    if not user_id or provider.provider_code == "local":
        return None
    config = provider.user_identity_forward
    if not config or not config.get("enabled"):
        return None
    field = config.get("field")
    if not field:
        return None
    value = (config.get("prefix") or "") + hashlib.sha256(str(user_id).encode()).hexdigest()
    max_len = config.get("max_len")
    if max_len and len(value) > max_len:
        value = value[:max_len]
    return field, value


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
