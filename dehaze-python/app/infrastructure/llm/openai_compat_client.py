"""OpenAI 兼容协议流式对话客户端"""

import json
from collections.abc import AsyncGenerator

from app.infrastructure.llm.model_client import LlmStreamChunk, build_auth_headers


class OpenAiCompatClient:
    """OpenAI 兼容协议实现（兼容 /chat/completions 的流式接口）"""

    def __init__(self, client) -> None:
        self._client = client

    @staticmethod
    def _convert_messages(messages: list[dict], system_prompt: str | None) -> list[dict]:
        """将内部消息列表转换为 OpenAI 兼容格式。

        消息 dict 携带 role/content，并可能携带 tool_call_id（role=tool）
        或 tool_calls（role=assistant），直接透传即可。
        """
        converted = []
        if system_prompt:
            converted.append({"role": "system", "content": system_prompt})
        converted.extend(messages)
        return converted

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
    ) -> AsyncGenerator[LlmStreamChunk, None]:
        """构建 OpenAI 兼容请求并解析 SSE 流，按 tool_call 索引聚合 function calling。

        OpenAI 兼容协议对稳定前缀自动缓存，无需主动干预。
        """
        payload = {
            "model": model.model_id,
            "messages": self._convert_messages(messages, system_prompt),
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens or model.max_output_tokens,
        }
        if tools is not None:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        url = provider.api_base_url.rstrip("/") + "/chat/completions"
        headers = build_auth_headers(provider, api_key)
        pending_tool_calls: dict[int, dict] = {}  # index -> {id, name, arguments}
        async with self._client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    yield LlmStreamChunk(type="text_delta", content=delta["content"])
                # 推理模型思考流：openai_compat 推理模型（如 deepseek-r1）
                # 经 reasoning_content 增量下发
                if delta.get("reasoning_content"):
                    yield LlmStreamChunk(type="thinking_delta", content=delta["reasoning_content"])
                for tool_call in delta.get("tool_calls") or []:
                    index = tool_call.get("index", 0)
                    if index not in pending_tool_calls:
                        fn = tool_call.get("function") or {}
                        pending_tool_calls[index] = {
                            "id": tool_call.get("id", ""),
                            "name": fn.get("name", ""),
                            "arguments": "",
                        }
                        yield LlmStreamChunk(
                            type="tool_call_start",
                            tool_call_id=pending_tool_calls[index]["id"],
                            tool_call_name=pending_tool_calls[index]["name"],
                        )
                    tc = pending_tool_calls[index]
                    fn = tool_call.get("function") or {}
                    if fn.get("id") and not tc["id"]:
                        tc["id"] = fn["id"]
                    if fn.get("name") and not tc["name"]:
                        tc["name"] = fn["name"]
                    arguments = fn.get("arguments")
                    if arguments:
                        tc["arguments"] += arguments
                        yield LlmStreamChunk(type="tool_call_delta", content=arguments)
                if chunk.get("usage"):
                    yield LlmStreamChunk(type="done", usage=chunk["usage"])
            # 流式结束：对每个未完成的 tool_call 发 tool_call_complete
            for index in sorted(pending_tool_calls):
                tc = pending_tool_calls[index]
                yield LlmStreamChunk(
                    type="tool_call_complete",
                    content=tc["arguments"],
                    tool_call_id=tc["id"],
                    tool_call_name=tc["name"],
                )
            pending_tool_calls.clear()
