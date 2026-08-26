"""DehazeChatModel：将 dehaze 的 LlmClient 适配为 LangChain BaseChatModel

deepagents 内部以 LangChain 协议调用 model（BaseChatModel.astream / ainvoke）。
本适配器把 LlmClient（多供应商、API Key 轮换、AES 解密）桥接为 BaseChatModel：

- _astream：逐块透出文本增量与工具调用增量，聚合 AIMessageChunk
- _agenerate：走流式聚合，返回完整 ChatResult（复用 _astream 避免双实现）
- usage metadata（token 统计）挂到 AIMessage.response_metadata，供计费结算透出

所有对话消息统一转为 LlmClient 所需的 OpenAI 兼容 dict，供应商协议差异
（openai_compat / anthropic）由 LlmClient 内部处理。
"""

import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.database import get_db_session
from app.infrastructure.llm.call.llm_client import llm_client

logger = logging.getLogger(__name__)


def _langchain_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """将 LangChain 消息转为 LlmClient 需要的 OpenAI 兼容 dict。

    tool_call 以 role=assistant + tool_calls 表达；tool 结果以 role=tool +
    tool_call_id 表达，与旧自研图使用的内部消息格式一致，LlmClient 原样透传。
    """
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AIMessage):
        tool_calls = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["args"]),
                },
            }
            for tc in message.tool_calls
        ]
        return {
            "role": "assistant",
            "content": message.content or "",
            **(tool_calls and {"tool_calls": tool_calls} or {}),
        }
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    # 兜底：未知类型按用户消息处理
    return {"role": "user", "content": message.content}


def _tools_to_openai(tools: list[BaseTool | dict[str, Any]]) -> list[dict]:
    """将 LangChain 工具列表转为 OpenAI Function 定义（LlmClient 需要的格式）。"""
    result = []
    for tool in tools:
        if isinstance(tool, dict):
            result.append(tool)
            continue
        spec = convert_to_openai_tool(tool)
        # convert_to_openai_tool 返回 {type, function} 或直接 {function}，转统一格式
        if "function" in spec:
            result.append(spec)
        else:
            result.append({"type": "function", "function": spec})
    return result


class DehazeChatModel(BaseChatModel):
    """包装 LlmClient 的 LangChain ChatModel 适配器。

    每次调用内部通过 get_db_session 获取 db，redis 由 llm_client.stream_chat
    编排层自取；本层只聚合流式结果。
    """

    model: str
    # 最近一次调用的 usage（token 统计），供 _agenerate 附加到最终消息
    _last_usage: dict[str, Any] = {}
    # 最近一次调用的实际路由归因（actual model/provider/key/latency/request_id），供计费透出
    _last_call_meta: dict[str, Any] = {}
    # deepagents 通过 bind_tools 注入工具定义与 tool_choice
    _bound_tools: list[Any] | None = None
    _bound_tool_choice: str | None = None

    @property
    def _llm_type(self) -> str:
        return "dehaze_llm_client"

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        """接受 deepagents 绑定的工具定义，返回 self（tools 在调用时读取）。

        deepagents 通过 bind_tools 注入工具，_astream/_agenerate 读取
        _bound_tools 传给 LlmClient。
        """
        self._bound_tools = list(tools)
        self._bound_tool_choice = tool_choice
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("DehazeChatModel 仅支持异步调用（ainvoke/astream）")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        chunk = AIMessageChunk(content="")
        tool_call_chunks = []
        thinking_parts = []
        async for c in self._astream(messages, stop, run_manager, **kwargs):
            chunk = chunk + c.message
            if c.message.tool_call_chunks:
                tool_call_chunks.extend(c.message.tool_call_chunks)
            thinking = c.message.additional_kwargs.get("thinking")
            if thinking:
                thinking_parts.append(thinking)
        final_message = AIMessage(
            content=chunk.content,
            response_metadata={
                "usage": dict(self._last_usage or {}),
                "call_meta": dict(self._last_call_meta or {}),
            },
        )
        if thinking_parts:
            final_message.additional_kwargs["thinking"] = "".join(thinking_parts)
        if tool_call_chunks:
            final_message.tool_calls = [
                {
                    "name": tc["name"],
                    "args": json.loads(tc["args"] or "{}"),
                    "id": tc["id"],
                    "type": "tool_call",
                }
                for tc in tool_call_chunks
            ]
        return ChatResult(generations=[ChatGeneration(message=final_message)])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """流式调用 LlmClient，逐块产出 ChatGenerationChunk。

        文本增量直接产出；工具调用增量聚合为完整 tool_call 后一次性产出；
        done 事件的 usage 写入 response_metadata。
        """
        model_id = self.model
        converted_messages = [_langchain_message_to_dict(m) for m in messages]
        tools: list[dict] | None = (
            _tools_to_openai(self._bound_tools or kwargs.get("tools") or []) or None
        )
        tool_choice: str | None = self._bound_tool_choice or kwargs.get("tool_choice")
        temperature = float(kwargs.get("temperature", 0.7))
        max_tokens = kwargs.get("max_tokens")

        usage: dict = {}
        call_meta: dict = {}
        pending_tool_calls: dict[int, dict] = {}
        # 思考流累积：LangChain 无标准 thinking 字段，经 additional_kwargs["thinking"]
        # 逐块透出，供 SseEventConverter 识别为思考双流
        thinking_accumulated = ""
        async with get_db_session() as db:
            async for chunk in llm_client.stream_chat(
                db,
                model_id,
                converted_messages,
                system_prompt=None,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                on_route_result=call_meta.update,
            ):
                if chunk.type == "text_delta":
                    yield ChatGenerationChunk(message=AIMessageChunk(content=chunk.content))
                elif chunk.type == "thinking_delta":
                    thinking_accumulated += chunk.content
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content="",
                            additional_kwargs={"thinking": chunk.content},
                        )
                    )
                elif chunk.type == "tool_call_start":
                    pending_tool_calls[chunk.tool_call_id] = {
                        "id": chunk.tool_call_id,
                        "name": chunk.tool_call_name,
                        "arguments": "",
                    }
                elif chunk.type == "tool_call_delta":
                    # 流式参数片段暂存，待 complete 时一次性产出
                    if chunk.tool_call_id in pending_tool_calls:
                        pending_tool_calls[chunk.tool_call_id]["arguments"] += chunk.content
                elif chunk.type == "tool_call_complete":
                    pending = pending_tool_calls.pop(chunk.tool_call_id, None)
                    if pending is not None:
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(
                                content="",
                                tool_call_chunks=[
                                    {
                                        "name": pending["name"],
                                        "args": pending["arguments"],
                                        "id": pending["id"],
                                        "index": None,
                                        "type": "tool_call_chunk",
                                    }
                                ],
                            )
                        )
                elif chunk.type == "done":
                    usage = chunk.usage or {}
        # 记录 usage 与实际路由归因，供 _agenerate 附加到最终消息（计费结算透出）
        self._last_usage = usage
        self._last_call_meta = call_meta
