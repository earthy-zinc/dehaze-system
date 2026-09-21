"""GuardrailMiddleware：deepagents 输入/输出护栏

设计文档 §8.5 安全护栏，运行时实时拦截异常输入与不合规输出，与静态文件系统
权限（§8.1）互补，构成"授权 + 拦截"双层防护。

拦截点：
- abefore_model：Prompt 注入防护（用户输入 + 工具结果 + 记忆注入通道）、敏感话题过滤
- awrap_tool_call：越权查询检测（阻止访问未授权的 MCP 命名空间）
- aafter_agent：敏感信息脱敏（身份证/手机号/密钥等 PII 与凭据）

开关与参数来自三级合并后的 config.guardrails；命中记录 guardrail 审计日志。
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ToolCallRequest,
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from app.service.ai.middleware.dehaze_hooks_middleware import DehazeAgentState
from app.service.ai.service import trace_collector
from app.utils.pii import mask_pii

logger = logging.getLogger(__name__)

# Prompt 注入关键词（试图篡改系统提示词/越权指令）
_INJECTION_KEYWORDS = (
    "忽略系统提示词",
    "忽略之前的指令",
    "ignore your system prompt",
    "ignore all previous instructions",
    "你是管理员",
    "system:",
)
# 敏感话题关键词（按业务场景屏蔽无关或不当请求）
_SENSITIVE_TOPIC_KEYWORDS = (
    "如何入侵",
    "如何破解",
    "制造炸弹",
    "获取他人隐私",
)
# 历史工具结果折叠进 assistant 消息时的标记（上下文组装层写入）
_TOOL_RESULT_MARKER = "[工具调用结果]"
# 元工具中可执行任意网关工具的入口：授权判定必须落到其目标工具名
_MCP_EXECUTE_TOOL = "mcp_execute_tool"


def _first_keyword(keywords: tuple[str, ...], texts: list[str]) -> str | None:
    """返回首个命中的关键词，无命中返回 None。"""
    for text in texts:
        lowered = text.lower()
        for kw in keywords:
            if kw.lower() in lowered:
                return kw
    return None


def _last_user_text(messages: list[Any]) -> str:
    for m in reversed(messages):
        if getattr(m, "type", "") == "human" and getattr(m, "content", None):
            return str(m.content)
    return ""


def _injected_channel_texts(messages: list[Any]) -> list[str]:
    """非用户直输通道的文本：工具结果与记忆注入块。

    工具结果（web 检索/MCP 返回）与长期记忆都由外部内容填充，是不受用户控制的
    注入面，必须与用户输入同等检测，否则关键词黑名单只对直输通道生效。
    """
    texts: list[str] = []
    for m in messages:
        mtype = getattr(m, "type", "")
        content = getattr(m, "content", None)
        if not content:
            continue
        text = str(content)
        is_tool_result = mtype == "tool" or (mtype == "ai" and _TOOL_RESULT_MARKER in text)
        if is_tool_result or mtype == "system":
            texts.append(text)
    return texts


def _blocked(content: str) -> dict[str, Any]:
    return {
        "messages": [
            AIMessage(content=content, response_metadata={"stop_reason": "guardrail_blocked"})
        ]
    }


class GuardrailMiddleware(AgentMiddleware):
    """deepagents 护栏中间件。

    Args:
        guardrails: 三级合并后的护栏配置（{prompt_injection:{enabled}, ...}）。
        allowed_mcp_namespaces: Agent 授权的 MCP 命名空间列表（空表示无命名空间工具）。
    """

    state_schema = DehazeAgentState

    def __init__(self, guardrails: dict[str, Any], allowed_mcp_namespaces: list[str]) -> None:
        self.guardrails = guardrails or {}
        self.allowed_mcp_namespaces = allowed_mcp_namespaces or []

    def _enabled(self, name: str) -> bool:
        return bool((self.guardrails.get(name) or {}).get("enabled", True))

    def _log_hit(self, rule: str, detail: str) -> None:
        logger.warning("Guardrail hit rule=%s detail=%s", rule, detail)
        # 可观测性：护栏命中写入过程链上下文事件（旁路：无采集器时静默跳过）
        collector = trace_collector.current()
        if collector is not None:
            collector.record_event(event="guardrail", rule=rule, detail=detail)

    def _authorized(self, tool_name: str) -> bool:
        """工具是否落在授权命名空间内（命名空间可为多段，按 <namespace>_ 前缀判定）。"""
        return any(tool_name.startswith(ns + "_") for ns in self.allowed_mcp_namespaces)

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        last_user = _last_user_text(messages)

        if self._enabled("prompt_injection"):
            texts = [last_user, *_injected_channel_texts(messages)]
            kw = _first_keyword(_INJECTION_KEYWORDS, texts)
            if kw:
                self._log_hit("prompt_injection", kw)
                return _blocked("检测到疑似 Prompt 注入指令，已拒绝处理。")
            self._warn_conversation_prompt(state)
        if self._enabled("sensitive_topic"):
            kw = _first_keyword(_SENSITIVE_TOPIC_KEYWORDS, [last_user])
            if kw:
                self._log_hit("sensitive_topic", kw)
                return _blocked("该话题不在服务范围内，无法处理。")
        return None

    def _warn_conversation_prompt(self, state: Any) -> None:
        """会话提示词拼接通道：命中注入关键词仅告警，不拦截。

        会话提示词由会话配置方提供，命中只说明配置内容可疑（拦截会直接让会话不可用），
        故留痕供审计而非阻断。
        """
        prompt = state.get("conversation_prompt") if isinstance(state, dict) else None
        if not prompt:
            return
        kw = _first_keyword(_INJECTION_KEYWORDS, [str(prompt)])
        if kw:
            self._log_hit("prompt_injection_warn", kw)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        # 越权查询检测：命名空间 MCP 工具（langchain-mcp-adapters 装载，命名
        # <namespace>_<tool>）仅在授权命名空间内放行。网关元工具 mcp_lookup_tool
        # 只做检索，不参与校验；mcp_execute_tool 以 M2M 密钥可调用网关任意工具，
        # 必须按其目标工具名校验，否则 Agent 的命名空间授权形同虚设。
        if not (self._enabled("unauthorized_access") and self.allowed_mcp_namespaces):
            return await handler(request)

        tool_call = request.tool_call or {}
        tool_name = tool_call.get("name", "")
        tool_call_id = tool_call.get("id", "")
        if tool_name == _MCP_EXECUTE_TOOL:
            target = (tool_call.get("args") or {}).get("tool_name") or ""
            if not target:
                self._log_hit("unauthorized_access", "tool=mcp_execute_tool target=<missing>")
                return ToolMessage(
                    content="工具调用被拦截：mcp_execute_tool 缺少目标工具名",
                    tool_call_id=tool_call_id,
                )
        elif tool_name.startswith("mcp_"):
            return await handler(request)
        else:
            target = tool_name

        if target and not self._authorized(target):
            self._log_hit("unauthorized_access", f"tool={target}")
            return ToolMessage(
                content="工具调用被拦截：当前 Agent 无权访问该 MCP 命名空间",
                tool_call_id=tool_call_id,
            )
        return await handler(request)

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if not self._enabled("pii_mask"):
            return None
        masked = False
        # 落库通道脱敏：全部 AI 消息都要处理（流式出口另由 SseEventConverter 脱敏）
        for m in state.get("messages") or []:
            if isinstance(m, AIMessage) and getattr(m, "content", None):
                new_content = mask_pii(str(m.content))
                if new_content != m.content:
                    m.content = new_content
                    masked = True
        if masked:
            collector = trace_collector.current()
            if collector is not None:
                collector.record_event(event="guardrail", rule="pii_mask", detail="masked")
        return None
