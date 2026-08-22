"""Anthropic 协议流式对话客户端"""

import json
from collections.abc import AsyncGenerator

from app.infrastructure.llm.model_client import LlmStreamChunk, build_auth_headers


class AnthropicClient:
    """Anthropic 原生协议实现（/messages 流式接口，含 Prompt Caching 注入）"""

    def __init__(self, client) -> None:
        self._client = client

    @staticmethod
    def _convert_messages_anthropic(messages: list[dict]) -> list[dict]:
        """将内部消息列表转换为 Anthropic 原生格式。

        - role=tool → role=user + tool_result 内容块
        - role=assistant 携带 tool_calls → 追加 tool_use 内容块
        """
        converted = []
        for msg in messages:
            role = msg.get("role")
            if role == "tool":
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id", ""),
                                "content": msg.get("content", ""),
                            }
                        ],
                    }
                )
            elif role == "assistant":
                content = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    try:
                        input_obj = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        input_obj = {}
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": input_obj,
                        }
                    )
                if content:
                    converted.append({"role": "assistant", "content": content})
            else:
                converted.append({"role": role, "content": msg.get("content", "")})
        return converted

    @staticmethod
    def _convert_tools_anthropic(tools: list[dict]) -> list[dict]:
        """将 OpenAI Function 工具定义转换为 Anthropic 原生格式"""
        converted = []
        for tool in tools:
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            converted.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
            )
        return converted

    @staticmethod
    def _convert_tool_choice_anthropic(tool_choice: str | None) -> dict | None:
        """将 OpenAI tool_choice 字符串转换为 Anthropic 原生格式"""
        if tool_choice is None:
            return None
        mapping = {"auto": "auto", "none": "none", "required": "any", "any": "any"}
        if tool_choice in mapping:
            return {"type": mapping[tool_choice]}
        # 指定具体工具名
        return {"type": "tool", "name": tool_choice}

    def _should_cache(self, model) -> bool:
        """是否启用 Prompt Caching（anthropic 需主动注入 cache_control）"""
        return bool(model.supports_prompt_cache and model.prompt_cache_prefix_len > 0)

    async def stream_chat(
        self,
        provider,
        api_key: str,
        model,
        messages: list[dict],
        system_prompt: str | None,
        max_tokens: int | None,
        tools: list[dict] | None,
        tool_choice: str | None,
        temperature: float = 0.7,  # noqa: ARG002 anthropic 无需显式传温度
    ) -> AsyncGenerator[LlmStreamChunk, None]:
        """构建 Anthropic 原生请求并解析 SSE 流，聚合 tool_use 内容块为三段式 tool_call 事件。

        模型启用 Prompt Caching 时，对稳定前缀（system + 最后一个工具定义）
        注入 cache_control，命中缓存按 cached_rate 计费。
        """
        payload = {
            "model": model.model_id,
            "messages": self._convert_messages_anthropic(messages),
            "stream": True,
            "max_tokens": max_tokens or model.max_output_tokens,
        }
        cache = self._should_cache(model)
        if system_prompt:
            if cache:
                # 稳定前缀：system 转内容块并标记 cache_control
                payload["system"] = [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                payload["system"] = system_prompt
        if tools is not None:
            anthropic_tools = self._convert_tools_anthropic(tools)
            if cache and anthropic_tools:
                # 工具定义为稳定前缀，对最后一个工具注入 cache_control
                anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
            payload["tools"] = anthropic_tools
            anthropic_choice = self._convert_tool_choice_anthropic(tool_choice)
            if anthropic_choice is not None:
                payload["tool_choice"] = anthropic_choice
        url = provider.api_base_url.rstrip("/") + "/messages"
        headers = build_auth_headers(provider, api_key)
        headers.setdefault("anthropic-version", "2023-06-01")
        usage: dict = {}
        pending: dict[int, dict] = {}  # index -> {id, name, arguments}
        async with self._client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "message_start":
                    usage.update(event.get("message", {}).get("usage") or {})
                elif etype == "content_block_start":
                    cb = event.get("content_block") or {}
                    if cb.get("type") == "tool_use":
                        index = event.get("index", 0)
                        pending[index] = {
                            "id": cb.get("id", ""),
                            "name": cb.get("name", ""),
                            "arguments": "",
                        }
                        yield LlmStreamChunk(
                            type="tool_call_start",
                            tool_call_id=pending[index]["id"],
                            tool_call_name=pending[index]["name"],
                        )
                    elif cb.get("type") == "thinking":
                        # 推理模型思考流：initial thinking 文本一次下发，signature 丢弃
                        thinking = cb.get("thinking") or ""
                        if thinking:
                            yield LlmStreamChunk(type="thinking_delta", content=thinking)
                elif etype == "content_block_delta":
                    delta = event.get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        yield LlmStreamChunk(type="text_delta", content=delta["text"])
                    elif delta.get("type") == "thinking_delta" and delta.get("thinking"):
                        yield LlmStreamChunk(type="thinking_delta", content=delta["thinking"])
                    elif delta.get("type") == "input_json_delta":
                        index = event.get("index", 0)
                        partial = delta.get("partial_json", "")
                        if index in pending:
                            pending[index]["arguments"] += partial
                            yield LlmStreamChunk(type="tool_call_delta", content=partial)
                elif etype == "content_block_stop":
                    index = event.get("index", 0)
                    if index in pending:
                        tc = pending.pop(index)
                        yield LlmStreamChunk(
                            type="tool_call_complete",
                            content=tc["arguments"],
                            tool_call_id=tc["id"],
                            tool_call_name=tc["name"],
                        )
                elif etype == "message_delta":
                    usage.update(event.get("usage") or {})
        yield LlmStreamChunk(type="done", usage=usage)
