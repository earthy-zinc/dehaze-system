"""GuardrailMiddleware：deepagents 输入/输出护栏

设计文档 §8.5 安全护栏，运行时实时拦截异常输入与不合规输出，与静态文件系统
权限（§8.1）互补，构成"授权 + 拦截"双层防护。

拦截点：
- abefore_model：Prompt 注入防护、敏感话题过滤（输入）
- awrap_tool_call：越权查询检测（阻止访问未授权的 MCP 命名空间）
- aafter_agent：敏感信息脱敏（身份证/手机号/密钥等 PII 与凭据）

开关与参数来自三级合并后的 config.guardrails；命中记录 guardrail 审计日志。
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
)
from langchain_core.messages import AIMessage, ToolMessage

from app.service.ai.dehaze_hooks_middleware import DehazeAgentState
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
        # TODO: 接入 guardrail 审计日志持久化（MongoDB 审计），当前记录应用日志便于观测误拦率
        logger.warning("Guardrail hit rule=%s detail=%s", rule, detail)

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        # 取最近一条用户消息做输入护栏检查
        last_user = ""
        for m in reversed(messages):
            if getattr(m, "type", "") == "human" and getattr(m, "content", None):
                last_user = str(m.content)
                break
        if not last_user:
            return None

        if self._enabled("prompt_injection"):
            for kw in _INJECTION_KEYWORDS:
                if kw.lower() in last_user.lower():
                    self._log_hit("prompt_injection", kw)
                    return {
                        "messages": [
                            AIMessage(
                                content="检测到疑似 Prompt 注入指令，已拒绝处理。",
                                response_metadata={"stop_reason": "guardrail_blocked"},
                            )
                        ]
                    }
        if self._enabled("sensitive_topic"):
            for kw in _SENSITIVE_TOPIC_KEYWORDS:
                if kw in last_user:
                    self._log_hit("sensitive_topic", kw)
                    return {
                        "messages": [
                            AIMessage(
                                content="该话题不在服务范围内，无法处理。",
                                response_metadata={"stop_reason": "guardrail_blocked"},
                            )
                        ]
                    }
        return None

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        # 越权查询检测：命名空间 MCP 工具（langchain-mcp-adapters 装载，命名
        # <namespace>_<tool>）仅在授权命名空间内放行。内建网关工具
        # （mcp_lookup_tool / mcp_execute_tool）是普通业务工具，不参与此校验。
        if self._enabled("unauthorized_access") and self.allowed_mcp_namespaces:
            tool_call = request.tool_call or {}
            tool_name = tool_call.get("name", "")
            if tool_name and not tool_name.startswith("mcp_"):
                namespace = tool_name.split("_", 1)[0]
                if namespace and namespace not in self.allowed_mcp_namespaces:
                    self._log_hit("unauthorized_access", f"tool={tool_name}")
                    return ToolMessage(
                        content="工具调用被拦截：当前 Agent 无权访问该 MCP 命名空间",
                        tool_call_id=tool_call.get("id", ""),
                    )
        return await handler(request)

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if not self._enabled("pii_mask"):
            return None
        messages = state.get("messages") or []
        # 对最后一条 AI 消息内容做 PII 脱敏
        masked = False
        for m in reversed(messages):
            if isinstance(m, AIMessage) and getattr(m, "content", None):
                new_content = mask_pii(str(m.content))
                if new_content != m.content:
                    m.content = new_content
                    masked = True
            if masked:
                break
        return None
