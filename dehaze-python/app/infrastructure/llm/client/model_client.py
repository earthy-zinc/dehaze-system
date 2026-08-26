"""LLM 协议客户端统一接口与工厂

按 sys_ai_provider.protocol_type 分发到具体协议实现（openai_compat / anthropic）。
LlmClient 只依赖此接口做编排；协议层共享类型与工具在 common 中，具体协议实现
亦从 common 引用，避免与 model_client 形成 import 环。
"""

from collections.abc import AsyncGenerator
from typing import Protocol

import httpx

from app.infrastructure.llm.common import (
    LlmStreamChunk,
    PROTOCOL_ANTHROPIC,
    PROTOCOL_OPENAI_COMPAT,
)


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
        from app.infrastructure.llm.client.anthropic_client import AnthropicClient

        return AnthropicClient(client)
    from app.infrastructure.llm.client.openai_compat_client import OpenAiCompatClient

    return OpenAiCompatClient(client)
