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
from contextvars import ContextVar
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
from app.models.base import get_current_user_id

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
        result: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result
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


# 模型实例按 (agent_id, version_no, model_id) 缓存跨会话复用，单次调用的 usage、
# 路由归因与绑定工具若存实例字段，并发 run 会互相覆盖（计费/成本归因串到他用户），
# 故按 asyncio 任务隔离（bind_tools 与随后的 ainvoke 在同一节点任务内执行）
_usage_var: ContextVar[dict | None] = ContextVar("dehaze_chat_model_usage", default=None)
_call_meta_var: ContextVar[dict | None] = ContextVar("dehaze_chat_model_call_meta", default=None)
_bound_tools_var: ContextVar[list | None] = ContextVar(
    "dehaze_chat_model_bound_tools", default=None
)
_bound_tool_choice_var: ContextVar[str | None] = ContextVar(
    "dehaze_chat_model_bound_tool_choice", default=None
)


class DehazeChatModel(BaseChatModel):
    """包装 LlmClient 的 LangChain ChatModel 适配器。

    每次调用内部通过 get_db_session 获取 db，redis 由 llm_client.stream_chat
    编排层自取；本层只聚合流式结果。
    """

    model: str

    @property
    def _llm_type(self) -> str:
        return "dehaze_llm_client"

    @property
    def _last_usage(self) -> dict[str, Any]:
        """本任务最近一次调用的 usage（token 统计），供计费结算透出"""
        return _usage_var.get() or {}

    @property
    def _last_call_meta(self) -> dict[str, Any]:
        """本任务最近一次调用的实际路由归因，供成本归因透出"""
        return _call_meta_var.get() or {}

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        """接受 deepagents 绑定的工具定义，返回 self（tools 在调用时读取）。

        deepagents 通过 bind_tools 注入工具，_astream/_agenerate 读取
        本任务绑定的工具定义传给 LlmClient。
        """
        _bound_tools_var.set(list(tools))
        _bound_tool_choice_var.set(tool_choice)
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
            message = c.message
            chunk = chunk + message
            if isinstance(message, AIMessageChunk) and message.tool_call_chunks:
                tool_call_chunks.extend(message.tool_call_chunks)
            thinking = message.additional_kwargs.get("thinking")
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
            _tools_to_openai(_bound_tools_var.get() or kwargs.get("tools") or []) or None
        )
        tool_choice: str | None = _bound_tool_choice_var.get() or kwargs.get("tool_choice")
        temperature = float(kwargs.get("temperature", 0.7))
        max_tokens = kwargs.get("max_tokens")

        usage: dict = {}
        call_meta: dict = {}
        pending_tool_calls: dict[str, dict] = {}
        # 工具调用产出序号：tool_call_chunks 的 index 按产出顺序编号（LangChain
        # 多工具并行合并时按 index 匹配增量），不能为 None
        tool_call_seq = 0
        # 思考流累积：LangChain 无标准 thinking 字段，经 additional_kwargs["thinking"]
        # 逐块透出，供 SseEventConverter 识别为思考双流
        thinking_accumulated = ""
        # 用户身份（供供应商透传）：图按 (agent_id, version, model) 跨用户缓存，
        # 实例无法携带用户标识，从请求上下文取（与 LlmClient 计费归因同源，
        # auth 依赖按请求设置）；无请求上下文（评测/A2A 临时会话）时为 None 不注入
        user_id = get_current_user_id()
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
                user_id=user_id,
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
                                        "index": tool_call_seq,
                                        "type": "tool_call_chunk",
                                    }
                                ],
                            )
                        )
                        tool_call_seq += 1
                elif chunk.type == "done":
                    usage = chunk.usage or {}
        # 记录 usage 与实际路由归因，供 _agenerate 附加到最终消息（计费结算透出）
        _usage_var.set(usage)
        _call_meta_var.set(call_meta)
