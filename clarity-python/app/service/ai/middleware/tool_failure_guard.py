"""连续失败保护中间件

设计文档 §2.3（能力扩展）：同一工具同一推理内连续失败达 3 次后，将该工具从当前
可用工具列表临时剔除并提示 LLM 更换能力；成功清零；新推理轮次重置计数。

实现要点：
- 计数按会话分桶（conversation_id，缺失时回退 msg_id/agent_id 的独立键，绝不共享
  同一个兜底桶）：中间件实例随图缓存共享（按 Agent 版本复用），多会话并发 run 时
  各会话的计数与禁用集互相隔离。
- 桶在会话每次 run 开始（abefore_agent）时重置，不写入图 state（不持久化）。
  会话数有限且桶极小（工具名计数），随图缓存重建整体释放，不做额外淘汰。
- awrap_tool_call 观察实际工具执行：异常或 ToolMessage.status="error" 记为失败，
  成功清零；连续失败达阈值后剔除并返回提示消息。
- awrap_model_call 将已剔除工具从本次模型请求的 tools 中过滤（request.override(tools=...)），
  使 LLM 不再被该工具诱导重试。
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from app.config import settings

logger = logging.getLogger(__name__)


def _tool_name(tool: Any) -> str | None:
    """从 BaseTool 或 dict 工具提取名称。"""
    if isinstance(tool, dict):
        name = tool.get("name")
        return name if isinstance(name, str) else None
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None


def _conv_key(state: Any) -> int | str:
    """提取隔离键：conversation_id 优先，缺失时退回 msg_id/agent_id 的独立键。

    conversation_id 缺失（如子图/单测直调）时若统一归入 0 号桶，互不相关的调用会
    共享连续失败计数并互相禁用工具，故按调用处唯一键隔离。
    """
    if isinstance(state, dict):
        conv = state.get("conversation_id")
        msg_id = state.get("message_id")
        agent_id = state.get("agent_id")
    else:
        conv = getattr(state, "conversation_id", None)
        msg_id = getattr(state, "message_id", None)
        agent_id = getattr(state, "agent_id", None)
    if conv:
        return conv
    if msg_id:
        return f"msg:{msg_id}"
    if agent_id:
        return f"agent:{agent_id}"
    return id(state)


class ToolFailureGuardMiddleware(AgentMiddleware):
    """同一工具连续失败达阈值后临时剔除并提示更换能力（按会话隔离）。"""

    def __init__(self, fail_limit: int | None = None) -> None:
        self.fail_limit = fail_limit or settings.AI_TOOL_CONSECUTIVE_FAIL_LIMIT
        # 隔离键 → {tool_name: 连续失败计数} / {tool_name: 禁用集合}
        self._fails: dict[int | str, dict[str, int]] = {}
        self._disabled: dict[int | str, set[str]] = {}

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        # 该会话新 run 开始：重置其连续失败计数与剔除集合（不影响其他并发会话）
        conv = _conv_key(state)
        self._fails.pop(conv, None)
        self._disabled.pop(conv, None)
        return None

    async def awrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        disabled = self._disabled.get(_conv_key(getattr(request, "state", request)))
        if disabled:
            filtered = [t for t in request.tools if _tool_name(t) not in disabled]
            request = request.override(tools=filtered)
        return await handler(request)

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        conv = _conv_key(getattr(request, "state", request))
        tool_call = request.tool_call or {}
        tool_name = tool_call.get("name", "")
        tool_call_id = tool_call.get("id", "")

        if tool_name in (self._disabled.get(conv) or ()):
            return ToolMessage(
                content="该工具已因连续失败被临时禁用，请更换其他能力完成此任务。",
                tool_call_id=tool_call_id,
                status="error",
            )

        try:
            result = await handler(request)
        except Exception as exc:
            logger.warning("工具 %s 调用异常: %s", tool_name, exc)
            return self._record_failure(conv, tool_name, tool_call_id, f"工具调用异常：{exc}")

        if isinstance(result, ToolMessage) and self._is_failure(result):
            return self._record_failure(
                conv, tool_name, tool_call_id, result.content, base_message=result
            )
        self._fails.setdefault(conv, {})[tool_name] = 0
        return result

    def _is_failure(self, message: ToolMessage) -> bool:
        """判定工具结果为失败：status=error 或工具执行失败（_dehaze_status=2）。

        status=3 为跳过（服务不可用等未真正执行），不是工具自身的连续失败，
        计入会让可用工具被误剔除。
        """
        if getattr(message, "status", "") == "error":
            return True
        return (getattr(message, "additional_kwargs", {}) or {}).get("_dehaze_status") == 2

    def _record_failure(
        self,
        conv: int | str,
        tool_name: str,
        tool_call_id: str,
        content: str | list[str | dict[Any, Any]],
        base_message: ToolMessage | None = None,
    ) -> ToolMessage:
        """累计连续失败，达阈值则剔除该工具并返回提示消息。"""
        count = self._fails.setdefault(conv, {}).get(tool_name, 0) + 1
        self._fails[conv][tool_name] = count
        if count >= self.fail_limit:
            self._disabled.setdefault(conv, set()).add(tool_name)
            logger.warning("会话 %s 工具 %s 连续失败 %s 次，已临时禁用", conv, tool_name, count)
            hint = f"{content}\n该工具连续失败已临时禁用，请更换能力。"
            if base_message is not None:
                return base_message.model_copy(update={"content": hint})
            return ToolMessage(content=hint, tool_call_id=tool_call_id, status="error")
        return (
            base_message
            if base_message is not None
            else ToolMessage(content=content, tool_call_id=tool_call_id, status="error")
        )
