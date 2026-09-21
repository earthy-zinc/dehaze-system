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
import json
import logging
import time
from collections.abc import Awaitable, Callable

from app.database import get_db_session
from app.repository.ai_conversation_repository import ai_conversation_repository
from app.service.ai.middleware.tool_recovery import classify_tool_error
from app.service.ai.service import trace_collector
from app.service.ai.service.memory_extraction import extract_memories, save_extracted_memories
from app.service.billing.billing_service import billing_service
from app.service.billing.estimate_service import estimate_service

logger = logging.getLogger(__name__)

# 钩子函数签名：接收当前 state（hooks 兼容 dict，含 AgentState 之外的运行期键），
# 返回 dict 表示中断/结果，None 表示继续
HookFunc = Callable[[dict], Awaitable[dict | None]]

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

    async def run_hooks(self, hook_point: str, state: dict) -> dict | None:
        """执行某个钩子点的所有钩子。

        前置钩子（before_*）返回非 None 表示中断，后续钩子不再执行。
        """
        for _, func in self._hooks[hook_point]:
            result = await func(state)
            if result is not None:
                return result
        return None


# 全局单例
agent_hooks = AgentHooks()


# ===== 内置钩子 =====


async def _step_limit_hook(state: dict) -> dict | None:
    """步数限制钩子（before_model）：超过最大步数强制终止"""
    if state.get("step_count", 0) >= state["max_steps"]:
        return {
            "final_response": "已达到最大推理步数限制，请简化需求或分多次对话",
            "stop_reason": "max_steps",
        }
    return None


async def _token_budget_hook(state: dict) -> dict | None:
    """Token 预算控制钩子（before_model）：超过预算上限强制终止"""
    if state.get("token_used", 0) >= state["token_budget"]:
        return {
            "final_response": "已达到 Token 预算上限",
            "stop_reason": "token_budget_exceeded",
        }
    return None


def _current_user_message(state: dict) -> str:
    """取当前用户消息内容（最后一个 user 消息），用于积分预估"""
    for msg in reversed(state.get("messages") or []):
        if msg.get("role") == "user" and msg.get("content"):
            return msg["content"]
    return ""


async def _billing_pre_charge_hook(state: dict) -> dict | None:
    """计费预扣钩子（before_agent）：欠费熔断 + 预估 + 配额/余额预扣。

    预扣成功 → 注入 billing_context 到 state 并继续；失败 → 返回中断数据阻断推理。
    子 Agent run 不预扣：整轮预算由主图预扣承担，子图完成后按实际用量实报实销
    （见 _billing_settle_hook 与 billing_service.settle_subagent）。
    """
    if state.get("is_subagent"):
        return None
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


async def _billing_budget_hook(state: dict) -> dict | None:
    """滚动预算钩子（before_model）：单步预估超剩余预算时中断"""
    if not state.get("billing_context"):
        return None
    async with get_db_session() as db:
        step_estimated = await estimate_service.estimate_step_credits(
            db, state.get("model_id", ""), state.get("messages") or []
        )
        return await billing_service.check_budget(state, step_estimated)


async def _billing_settle_hook(state: dict) -> dict | None:
    """实扣结算钩子（after_agent）：按实际用量差额退补 + 更新计费记录。

    主图（有 billing_context）：差额退补 + 更新预扣记录，行为与历史一致。
    子图（is_subagent，与主 run 共享 ctx，billing_context 可能存在——判定以
    is_subagent 为准，防止子图二次结算主图预扣）：实报实销——按实际用量直接
    扣减配额与余额，独立计费记录 bill_type=chat_subagent（归属主会话用户/消息）。
    降级时 actual_model 取实际路由归因（call_meta.model_id），并透传供应商/
    延迟/错误码/请求号用于成本归因。

    结算失败（如价格配置缺失）不中断 after_agent 链：回复与预扣已完成，
    后续钩子（trace 落库/记忆提取）不得被计费异常吞掉，错误记日志供对账。
    """
    try:
        if state.get("is_subagent"):
            user_id = state.get("user_id")
            conversation_id = state.get("conversation_id")
            message_id = state.get("message_id")
            model_id = state.get("model_id")
            if not (user_id and conversation_id and message_id and model_id):
                return None
            call_meta = state.get("call_meta") or {}
            async with get_db_session() as db:
                await billing_service.settle_subagent(
                    db,
                    user_id,
                    conversation_id,
                    message_id,
                    model_id,
                    call_meta.get("model_id"),
                    state.get("usage") or {},
                    provider_id=call_meta.get("provider_id"),
                    latency_ms=call_meta.get("latency_ms"),
                    error_code=call_meta.get("error_code"),
                    request_id=call_meta.get("request_id"),
                )
            return None
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
    except Exception:
        logger.error(
            "计费结算失败(预扣已落库，待对账): message_id=%s",
            state.get("message_id"),
            exc_info=True,
        )
    return None


# 异步后台任务引用，防止被垃圾回收
_pending_tasks: set[asyncio.Task] = set()


async def _memory_extraction_hook(state: dict) -> dict | None:
    """记忆提取钩子（after_agent）：异步触发，不阻塞主流程。

    子图 run 跳过：记忆口径归属主会话完整对话，子图消息源于任务描述，
    提取入长期记忆会污染用户画像。
    """
    if state.get("is_subagent"):
        return None
    user_id = state.get("user_id")
    model_id = state.get("model_id")
    messages = state.get("messages")
    conversation_id = state.get("conversation_id")
    if not user_id or not model_id or not messages or not conversation_id:
        return None

    async def _run() -> None:
        try:
            memories = await extract_memories(
                user_id, model_id, messages, conversation_id=conversation_id
            )
            await save_extracted_memories(user_id, memories)
        except Exception:
            logger.warning("Memory extraction failed", exc_info=True)

    task = asyncio.create_task(_run())
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)
    return None


async def _title_update_hook(state: dict) -> dict | None:
    """会话标题更新钩子（after_agent）：异步触发。

    会话无标题或标题为自动生成（title_source=auto）时，取首条 user+assistant 消息
    LLM 生成 ≤20 字标题并更新会话；复用会话域 AiConversationService._auto_generate_title。
    子图 run 跳过：标题归属主会话，子图消息不能作为主会话标题依据。
    """
    if state.get("is_subagent"):
        return None
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


async def _tool_recovery_hook(state: dict) -> dict | None:
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


# ===== 并行子 Agent 写冲突仲裁（before_tool）=====

# 单次让行等待上限（秒）：持有者一次工具调用耗时长于该值的场景（批量处理/异步等待）
# 由其自身超时或中断兜底，让行方不宜无限挂起
_WRITE_YIELD_TIMEOUT = 30.0

# 显式写文件的内置工具（deepagents 文件系统中间件，目标参数统一为 file_path）
_FILE_WRITE_TOOLS = ("write_file", "edit_file", "delete")
# MCP 网关元工具透传的后端写接口前缀（网关工具命名为 <method>_<path>）
_API_WRITE_PREFIXES = ("post_", "put_", "patch_", "delete_")


def write_resource_key(tool_name: str, args: dict | None) -> str | None:
    """识别工具调用的写入资源键（文件路径 / 工具目标键）；只读操作返回 None。

    沙箱执行（execute_code）的写入目标由任意命令决定，无法归约为稳定资源键，不纳入仲裁。
    """
    args = args or {}
    if tool_name in _FILE_WRITE_TOOLS:
        path = args.get("file_path")
        return f"file:{path}" if path else None
    if tool_name == "mcp_execute_tool":
        api = str(args.get("tool_name") or "")
        if api.startswith(_API_WRITE_PREFIXES):
            params = json.dumps(args.get("arguments") or {}, sort_keys=True, ensure_ascii=False)
            return f"api:{api}:{params}"
    return None


def _occupy_write(ctx: dict, resource: str, holder: dict) -> None:
    """登记写入持有者，并记录本 run 内该资源的写入来源（串行覆盖判定依据）"""
    ctx.setdefault("write_holders", {})[resource] = {
        **holder,
        "depth": 1,
        "released": asyncio.Event(),
    }
    ctx.setdefault("write_written_by", {})[resource] = holder["name"]


async def _wait_write_released(entry: dict, deadline: float) -> bool:
    """有界等待持有者释放；True 表示已释放（释放后可能被他人抢占，须重新判定）"""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    try:
        await asyncio.wait_for(entry["released"].wait(), timeout=remaining)
    except TimeoutError:
        # 有界让行超时（非异常）：返回"未释放"由调用方转成跳过/重试；
        # 留痕便于定位长耗时持有者（批量处理/异步等待超时）
        logger.debug("等待写入资源释放超时 [resource=%s]，按未释放处理", entry.get("name"))
        return False
    return True


def _conflict_retry_verdict(ctx: dict, state: dict, resource: str, holder_name: str) -> dict:
    """同优先级冲突：失败重试；重试次数耗尽升级为写冲突中断（提示回滚）"""
    retries = ctx.setdefault("write_conflict_retries", {})
    used = retries.get(resource, 0)
    if used >= (state.get("retry_max") or 2):
        return {
            "action": "interrupt",
            "reason": (
                f"资源 {resource} 的写入冲突重试 {used} 次仍未消解（持有者「{holder_name}」）"
            ),
            "previousWriter": holder_name,
        }
    retries[resource] = used + 1
    return {
        "action": "retry",
        "reason": f"同优先级「{holder_name}」正在写入资源 {resource}，请稍后重试",
    }


async def _write_conflict_hook(state: dict) -> dict | None:
    """写冲突仲裁钩子（before_tool）：并行子 Agent 对同一资源的写入按优先级仲裁。

    state 需携带 run_ctx / resource / holder（{name, priority, instance}）/ retry_max：
    - 低优先级后到 → 让行（等待持有者释放，超时跳过本次写入）
    - 同优先级 → 失败重试（重试耗尽升级为写冲突中断）
    - 高优先级后到 → 持有者已开始执行不可抢占，同样让行等待，超时按失败重试
    - 本 run 内该资源已被其他 Agent 写入过（串行覆盖）→ 中断提示回滚；
      用户确认覆盖后（overwrite_approved）不再重复提示

    获取成功返回 None；持有登记由调用方在工具结束时经 release_write_resource 释放。
    """
    ctx = state.get("run_ctx")
    resource = state.get("resource")
    holder = state.get("holder")
    if ctx is None or not resource or not holder:
        return None

    written_by = ctx.setdefault("write_written_by", {})
    holders = ctx.setdefault("write_holders", {})
    deadline = time.monotonic() + _WRITE_YIELD_TIMEOUT
    while True:
        entry = holders.get(resource)
        if entry is None:
            # 获取到空闲资源；本 run 内已被他人写过则先交用户裁决覆盖（串行覆盖）
            previous = written_by.get(resource)
            if previous and previous != holder["name"] and not state.get("overwrite_approved"):
                return {
                    "action": "interrupt",
                    "reason": f"资源 {resource} 已由「{previous}」写入，本次写入将覆盖其结果",
                    "previousWriter": previous,
                }
            _occupy_write(ctx, resource, holder)
            return None
        if entry["name"] == holder["name"] and entry["instance"] == holder["instance"]:
            entry["depth"] += 1
            return None
        if holder["priority"] == entry["priority"]:
            return _conflict_retry_verdict(ctx, state, resource, entry["name"])
        if not await _wait_write_released(entry, deadline):
            # 优先级数字小者优先：数字大者为低优先级，让行超时后跳过本次写入
            if holder["priority"] > entry["priority"]:
                return {
                    "action": "skip",
                    "reason": (
                        f"更高优先级的「{entry['name']}」正在写入资源 {resource}，本次写入已跳过"
                    ),
                }
            return _conflict_retry_verdict(ctx, state, resource, entry["name"])


def release_write_resource(ctx: dict, resource: str, holder: dict) -> None:
    """释放工具调用占用的写入资源（成功/异常/中断路径均须调用）。

    重入计数归零才真正让出并唤醒让行等待者；非同 holder 不释放，防误放导致仲裁失效。
    """
    holders = ctx.get("write_holders") or {}
    entry = holders.get(resource)
    if not entry or entry["name"] != holder["name"] or entry["instance"] != holder["instance"]:
        return
    entry["depth"] -= 1
    if entry["depth"] > 0:
        return
    entry["released"].set()
    holders.pop(resource, None)


async def _trace_settle_hook(state: dict) -> dict | None:
    """过程链落库钩子（after_agent）：聚合写入 sys_ai_trace（成功/失败均写）。

    消耗取计费口径 usage（含缓存命中与多模态归集），模型取实际路由归因；
    子图 run 跳过：子图与主图共享主消息采集器（ContextVar 继承），子图 LLM 调用
    明细已挂到主 trace，若在此结算会以子图用量覆盖主消息过程链并遮蔽主图后续调用；
    采集为旁路：任何失败仅告警，不影响计费/记忆提取等后续钩子。
    """
    if state.get("is_subagent"):
        return None
    collector = trace_collector.current()
    if collector is None:
        return None
    try:
        await collector.settle(
            status=trace_collector.TRACE_STATUS_SUCCESS,
            usage=state.get("usage"),
            step_count=state.get("step_count", 0),
            actual_model=(state.get("call_meta") or {}).get("model_id"),
        )
    except Exception:
        logger.warning("过程链记录写入失败 trace_id=%s", collector.trace_id, exc_info=True)
    return None


# 注册内置钩子
agent_hooks.register("before_agent", _billing_pre_charge_hook, priority=10)
agent_hooks.register("before_model", _step_limit_hook, priority=10)
agent_hooks.register("before_model", _token_budget_hook, priority=20)
agent_hooks.register("before_model", _billing_budget_hook, priority=30)
agent_hooks.register("before_tool", _write_conflict_hook, priority=10)
agent_hooks.register("after_tool", _tool_recovery_hook, priority=10)
agent_hooks.register("after_agent", _billing_settle_hook, priority=5)
agent_hooks.register("after_agent", _trace_settle_hook, priority=8)
agent_hooks.register("after_agent", _memory_extraction_hook, priority=10)
agent_hooks.register("after_agent", _title_update_hook, priority=20)
