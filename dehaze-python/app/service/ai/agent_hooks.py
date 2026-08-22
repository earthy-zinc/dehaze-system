"""AgentHooks 生命周期钩子框架

提供 Agent 推理全生命周期的钩子机制，支持在关键节点注入横切逻辑
（安全、审计、计费、记忆等），避免硬编码到推理流程中。

钩子点链路：
    before_agent → before_model → (LLM 调用) → after_model
    → (工具调用) → after_tool → after_agent

钩子按优先级升序执行；前置钩子（before_*）返回非 None 表示中断后续链路。
内置钩子：步数限制与 Token 预算控制（before_model）、配额预校验（before_agent）、
记忆提取与会话标题更新（after_agent）。
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from app.database import get_db_session
from app.repository.ai_conversation_repository import ai_conversation_repository
from app.service.ai.agent_state import AgentState
from app.service.ai.memory_extraction import extract_memories, save_extracted_memories
from app.service.ai.tool_recovery import classify_tool_error
from app.service.billing.billing_service import billing_service
from app.service.billing.estimate_service import estimate_service

logger = logging.getLogger(__name__)

# 钩子函数签名：接收当前 state，返回 dict 表示中断/结果，None 表示继续
HookFunc = Callable[[AgentState], Awaitable[dict | None]]

# 钩子点定义（顺序即推理生命周期顺序）
HOOK_POINTS = (
    "before_agent",
    "before_model",
    "after_model",
    "before_tool",
    "after_tool",
    "after_agent",
)


class AgentHooks:
    """Agent 推理生命周期钩子管理器"""

    def __init__(self) -> None:
        self._hooks: dict[str, list[tuple[int, HookFunc]]] = {point: [] for point in HOOK_POINTS}

    def register(self, hook_point: str, func: HookFunc, priority: int = 0) -> None:
        """注册钩子，按优先级升序执行（数字小者优先）"""
        if hook_point not in self._hooks:
            raise ValueError(f"未知钩子点: {hook_point}")
        self._hooks[hook_point].append((priority, func))
        self._hooks[hook_point].sort(key=lambda x: x[0])

    async def run_hooks(self, hook_point: str, state: AgentState) -> dict | None:
        """执行某个钩子点的所有钩子。

        前置钩子（before_*）返回非 None 表示中断，后续钩子不再执行。
        """
        for _, func in self._hooks[hook_point]:
            result = await func(state)
            if result is not None:
                return result  # 中断后续链路
        return None


# 全局单例
agent_hooks = AgentHooks()


# ===== 内置钩子 =====


async def _step_limit_hook(state: AgentState) -> dict | None:
    """步数限制钩子（before_model）：超过最大步数强制终止"""
    if state.get("step_count", 0) >= state["max_steps"]:
        return {
            "final_response": "已达到最大推理步数限制，请简化需求或分多次对话",
            "stop_reason": "max_steps",
        }
    return None


async def _token_budget_hook(state: AgentState) -> dict | None:
    """Token 预算控制钩子（before_model）：超过预算上限强制终止"""
    if state.get("token_used", 0) >= state["token_budget"]:
        return {
            "final_response": "已达到 Token 预算上限",
            "stop_reason": "token_budget_exceeded",
        }
    return None


def _current_user_message(state: AgentState) -> str:
    """取当前用户消息内容（最后一个 user 消息），用于积分预估"""
    for msg in reversed(state.get("messages") or []):
        if msg.get("role") == "user" and msg.get("content"):
            return msg["content"]
    return ""


async def _billing_pre_charge_hook(state: AgentState) -> dict | None:
    """计费预扣钩子（before_agent）：欠费熔断 + 预估 + 配额/余额预扣。

    预扣成功 → 注入 billing_context 到 state 并继续；失败 → 返回中断数据阻断推理。
    """
    user_id = state.get("user_id")
    conversation_id = state.get("conversation_id")
    message_id = state.get("message_id")
    model_id = state.get("model_id")
    if not user_id or not conversation_id or not message_id or not model_id:
        return None

    async with get_db_session() as db:
        result = await billing_service.pre_charge(
            db,
            user_id,
            conversation_id,
            message_id,
            _current_user_message(state),
            model_id,
        )
        if "billing_id" in result:  # 预扣成功
            state["billing_context"] = result
            return None
        return result  # 欠费/配额/余额不足，阻断推理


async def _billing_budget_hook(state: AgentState) -> dict | None:
    """滚动预算钩子（before_model）：单步预估超剩余预算时中断"""
    if not state.get("billing_context"):
        return None
    async with get_db_session() as db:
        step_estimated = await estimate_service.estimate_step_credits(
            db, state.get("model_id", ""), state.get("messages") or []
        )
        return await billing_service.check_budget(state, step_estimated)


async def _billing_settle_hook(state: AgentState) -> dict | None:
    """实扣结算钩子（after_agent）：按实际用量差额退补 + 更新计费记录。

    降级时 actual_model 取实际路由归因（call_meta.model_id），并透传供应商/
    延迟/错误码/请求号用于成本归因。
    """
    bc = state.get("billing_context")
    if not bc:
        return None
    model_id = state.get("model_id") or ""
    usage = state.get("usage") or {}
    call_meta = state.get("call_meta") or {}
    actual_model_id = call_meta.get("model_id")
    async with get_db_session() as db:
        await billing_service.settle(
            db,
            bc["user_id"],
            bc["conversation_id"],
            bc["message_id"],
            model_id,
            actual_model_id,
            usage,
            provider_id=call_meta.get("provider_id"),
            latency_ms=call_meta.get("latency_ms"),
            error_code=call_meta.get("error_code"),
            request_id=call_meta.get("request_id"),
        )
    return None


# 异步后台任务引用，防止被垃圾回收
_pending_tasks: set[asyncio.Task] = set()


async def _memory_extraction_hook(state: AgentState) -> dict | None:
    """记忆提取钩子（after_agent）：异步触发，不阻塞主流程"""
    user_id = state.get("user_id")
    model_id = state.get("model_id")
    messages = state.get("messages")
    if not user_id or not model_id or not messages:
        return None

    async def _run() -> None:
        try:
            memories = await extract_memories(user_id, model_id, messages)
            await save_extracted_memories(user_id, memories)
        except Exception:
            logger.warning("Memory extraction failed", exc_info=True)

    task = asyncio.create_task(_run())
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)
    return None


async def _title_update_hook(state: AgentState) -> dict | None:
    """会话标题更新钩子（after_agent）：异步触发。

    会话无标题或标题为自动生成（title_source=auto）时，取首条 user+assistant 消息
    LLM 生成 ≤20 字标题并更新会话；复用会话域 AiConversationService._auto_generate_title。
    """
    conversation_id = state.get("conversation_id")
    if not conversation_id:
        return None

    async with get_db_session() as db:
        conv = await ai_conversation_repository.get_by_id(db, conversation_id)
        if not conv or conv.deleted:
            return None
        if conv.title and conv.title != "新对话" and conv.title_source != "auto":
            return None

    # 组装首条 user+assistant 消息作为标题生成输入
    first_parts = []
    for msg in state.get("messages") or []:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            first_parts.append(msg["content"])
            if len(first_parts) >= 2:
                break
    if not first_parts:
        return None
    context_text = " ".join(first_parts)

    async def _run() -> None:
        # 延迟导入避免 agent_hooks → ai_conversation_service → deep_agent_builder 循环依赖
        from app.service.ai_conversation_service import ai_conversation_service

        try:
            await ai_conversation_service._auto_generate_title(conversation_id, context_text)
        except Exception:
            logger.warning("会话标题更新失败 conv_id=%s", conversation_id, exc_info=True)

    task = asyncio.create_task(_run())
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)
    return None


async def _tool_recovery_hook(state: AgentState) -> dict | None:
    """工具错误恢复钩子（after_tool）：按错误类型产出恢复动作（§6.4）。

    state 需携带 tool_name / tool_error / retry_count / retry_max：
    - 权限不足 → interrupt(type=confirm)
    - 参数错误且未超重试上限 → retry（把错误返回 LLM 修正参数）
    - 服务不可用 → skip（记录为 skipped）
    - 超时/不可恢复 → fail（记录失败原因）
    """
    exc = state.get("tool_error")
    if exc is None:
        return None
    action = classify_tool_error(exc)
    # 参数错误仅在重试次数未达上限时重试，否则降级为失败
    if action.action == "retry":
        if state.get("retry_count", 0) < state.get("retry_max", 2):
            return {"action": "retry", "reason": action.reason, "status": action.status}
        return {"action": "fail", "reason": f"重试次数耗尽: {action.reason}", "status": 2}
    return {"action": action.action, "reason": action.reason, "status": action.status}


# 注册内置钩子
agent_hooks.register("before_agent", _billing_pre_charge_hook, priority=10)
agent_hooks.register("before_model", _step_limit_hook, priority=10)
agent_hooks.register("before_model", _token_budget_hook, priority=20)
agent_hooks.register("before_model", _billing_budget_hook, priority=30)
agent_hooks.register("after_tool", _tool_recovery_hook, priority=10)
agent_hooks.register("after_agent", _billing_settle_hook, priority=5)
agent_hooks.register("after_agent", _memory_extraction_hook, priority=10)
agent_hooks.register("after_agent", _title_update_hook, priority=20)
