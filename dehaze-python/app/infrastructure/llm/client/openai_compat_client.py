"""OpenAI 兼容协议流式对话客户端"""

import json
from collections.abc import AsyncGenerator, Iterator

from app.infrastructure.llm.common import LlmStreamChunk, build_auth_headers

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class _ThinkSplitter:
    """跨块剥离 content 内嵌的思考块。

    部分 OpenAI 兼容模型（如 deepseek 推理系列）不走 reasoning_content 字段，
    而是把思考以 <think>...</think> 内嵌在 content 中，且标签可能被流式
    分块拆开，必须缓冲拼接后切分：思考段转 thinking_delta，正文转 text_delta。

    思考段采用缓冲式：进入 <think> 后内容暂存、等 </think> 到达时一次性下发
    thinking_delta（保证思考区完整展示）；若流结束仍未闭合（小模型对 /no_think
    不遵从的常见形态），flush 将滞留内容降级为 text_delta，避免正文被吞成空回复。
    """

    def __init__(self) -> None:
        self._in_think = False
        self._buf = ""

    @staticmethod
    def _partial_suffix_len(buf: str, tag: str) -> int:
        """buffer 尾部与标签前缀的最长重合长度（不完整标签需保留待下一块）"""
        for k in range(min(len(buf), len(tag) - 1), 0, -1):
            if buf.endswith(tag[:k]):
                return k
        return 0

    def feed(self, text: str) -> Iterator[tuple[str, str]]:
        self._buf += text
        while True:
            if self._in_think:
                # 思考态：缓冲等待闭合标签，内容不实时下发
                idx = self._buf.find(_THINK_CLOSE)
                if idx >= 0:
                    seg, self._buf = self._buf[:idx], self._buf[idx + len(_THINK_CLOSE) :]
                    if seg:
                        yield ("thinking", seg)
                    self._in_think = False
                    continue
                return
            # 正文态：实时下发；找到 <think> 则转入思考态
            idx = self._buf.find(_THINK_OPEN)
            if idx >= 0:
                seg, self._buf = self._buf[:idx], self._buf[idx + len(_THINK_OPEN) :]
                if seg:
                    yield ("text", seg)
                self._in_think = True
                continue
            cut = len(self._buf) - self._partial_suffix_len(self._buf, _THINK_OPEN)
            if cut > 0:
                seg, self._buf = self._buf[:cut], self._buf[cut:]
                yield ("text", seg)
            return

    def flush(self) -> Iterator[tuple[str, str]]:
        """流结束时输出缓冲残留。

        思考态残留（<think> 未闭合）降级为正文输出，保证模型输出内容不丢失。
        """
        if self._buf:
            seg, self._buf = self._buf, ""
            yield ("text", seg)


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
            # OpenAI 协议：请求流式 usage 统计（随最终 choices=[] 的 chunk 单独下发）
            "stream_options": {"include_usage": True},
            "temperature": temperature,
            "max_tokens": max_tokens or model.max_output_tokens,
        }
        # 模型配置的厂商私有请求参数（如阿里云 enable_thinking、OpenAI reasoning_effort）
        # 合并进请求体；仅补充核心键之外的键，模型/messages/stream 等由调用方控制
        for key, value in (model.extra_request_params or {}).items():
            if key not in payload:
                payload[key] = value
        if tools is not None:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        url = provider.api_base_url.rstrip("/") + "/chat/completions"
        headers = build_auth_headers(provider, api_key)
        pending_tool_calls: dict[int, dict] = {}  # index -> {id, name, arguments}
        think_splitter = _ThinkSplitter()
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
                # usage 随无 choices 的最终 chunk 单独下发（stream_options.include_usage），
                # 不能因无 choices 而跳过
                if not choices:
                    if chunk.get("usage"):
                        yield LlmStreamChunk(type="done", usage=chunk["usage"])
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    for kind, seg in think_splitter.feed(delta["content"]):
                        yield LlmStreamChunk(
                            type="thinking_delta" if kind == "thinking" else "text_delta",
                            content=seg,
                        )
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
            # 流式结束：输出切分器滞留的尾部内容，再对每个未完成的 tool_call 发 complete
            for kind, seg in think_splitter.flush():
                yield LlmStreamChunk(
                    type="thinking_delta" if kind == "thinking" else "text_delta",
                    content=seg,
                )
            for index in sorted(pending_tool_calls):
                tc = pending_tool_calls[index]
                yield LlmStreamChunk(
                    type="tool_call_complete",
                    content=tc["arguments"],
                    tool_call_id=tc["id"],
                    tool_call_name=tc["name"],
                )
            pending_tool_calls.clear()
