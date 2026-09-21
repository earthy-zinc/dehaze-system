"""Anthropic 协议流式对话客户端"""

import json
import logging
from collections.abc import AsyncGenerator

from app.infrastructure.llm.common import LlmStreamChunk, build_auth_headers
from app.service.ai.service.trace_collector import record_wire_request, record_wire_response

logger = logging.getLogger(__name__)


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
                        # 历史 tool_call 的 arguments 非合法 JSON（上游截断/损坏）：
                        # 不阻断本次请求构建，但必须留痕以定位数据来源（不记录参数内容）
                        logger.warning(
                            "tool_call arguments 非合法 JSON，按空对象处理: tool=%s, call_id=%s",
                            fn.get("name", ""),
                            tc.get("id", ""),
                        )
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
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
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
        temperature: float = 0.7,
        user_identity: tuple[str, str] | None = None,
    ) -> AsyncGenerator[LlmStreamChunk, None]:
        """构建 Anthropic 原生请求并解析 SSE 流，聚合 tool_use 内容块为三段式 tool_call 事件。

        模型启用 Prompt Caching 时，对稳定前缀（system + 最后一个工具定义）
        注入 cache_control，命中缓存按 cached 档位价计费
        （sys_ai_model_price，见 AI模型管理 §2.12）。
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
        # 用户身份透传：field 支持嵌套路径（如 metadata.user_id），中间层不存在时初始化；
        # 首段字段已存在（核心键）时不覆盖
        if user_identity is not None:
            field, value = user_identity
            parts = field.split(".")
            if parts[0] not in payload:
                node = payload
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = value
        url = provider.api_base_url.rstrip("/") + "/messages"
        headers = build_auth_headers(provider, api_key)
        headers.setdefault("anthropic-version", "2023-06-01")
        # wire 级原始报文采集（旁路）：上报实际发送的完整请求体（不存 headers/URL/API Key）
        record_wire_request(payload)
        usage: dict = {}
        pending: dict[int, dict] = {}  # index -> {id, name, arguments}
        # 流式聚合为等价非流式结构所需字段（响应原文取 provider 下发值）
        wire_id: str | None = None
        wire_model: str | None = None
        stop_reason: str | None = None
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        completed_tool_calls: list[dict] = []
        async with self._client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    # 非 JSON 的 data 分片（厂商扩展/被截断）：跳过但留痕，
                    # 避免流解析异常被完全掩盖（只记分片长度，不记分片内容）
                    logger.warning("SSE 分片无法解析为 JSON，已跳过: len=%d", len(data))
                    continue
                etype = event.get("type")
                if etype == "message_start":
                    msg = event.get("message", {})
                    if msg.get("id") and wire_id is None:
                        wire_id = msg["id"]
                    if msg.get("model") and wire_model is None:
                        wire_model = msg["model"]
                    usage.update(msg.get("usage") or {})
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
                            thinking_parts.append(thinking)
                            yield LlmStreamChunk(type="thinking_delta", content=thinking)
                elif etype == "content_block_delta":
                    delta = event.get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        text_parts.append(delta["text"])
                        yield LlmStreamChunk(type="text_delta", content=delta["text"])
                    elif delta.get("type") == "thinking_delta" and delta.get("thinking"):
                        thinking_parts.append(delta["thinking"])
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
                        completed_tool_calls.append(tc)
                        yield LlmStreamChunk(
                            type="tool_call_complete",
                            content=tc["arguments"],
                            tool_call_id=tc["id"],
                            tool_call_name=tc["name"],
                        )
                elif etype == "message_delta":
                    delta = event.get("delta") or {}
                    if delta.get("stop_reason"):
                        stop_reason = delta["stop_reason"]
                    usage.update(event.get("usage") or {})
        # 流式结束：聚合等价非流式响应结构上报（tool_calls arguments 为原文）
        message: dict = {"role": "assistant", "content": "".join(text_parts) or None}
        if thinking_parts:
            message["thinking"] = "".join(thinking_parts)
        if completed_tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in completed_tool_calls
            ]
        record_wire_response(
            {
                "id": wire_id,
                "model": wire_model or payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": {
                            "end_turn": "stop",
                            "stop_sequence": "stop",
                            "max_tokens": "length",
                            "tool_use": "tool_calls",
                        }.get(stop_reason or ""),
                        "message": message,
                    }
                ],
                "usage": usage or None,
            }
        )
        yield LlmStreamChunk(type="done", usage=usage)
