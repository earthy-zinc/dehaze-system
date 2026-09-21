"""DehazeHooksMiddleware：将 dehaze 的 AgentHooks 适配为 deepagents AgentMiddleware

设计文档 §3.3 Hooks 适配方案：
- before_agent / after_agent → abefore_agent / aafter_agent（计费预扣 / 实扣结算、记忆提取）
- before_model 步数限制 / Token 预算 → awrap_model_call 在调用模型前短路拦截
- 原 before_model / after_model / before_tool / after_tool 钩子框架保留在
  agent_hooks 中，通过 awrap_model_call 与 awrap_tool_call 桥接调用。

deepagents state 的 messages 是 LangChain BaseMessage 列表，与旧 AgentState 的
dict 列表不同，故将 hooks 所需字段映射为 AgentState 兼容 dict 后再调 agent_hooks。
计费上下文（billing_context）随推理链路经 ctx 贯穿，避免跨钩子点丢失。
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, NotRequired

from deepagents.graph import DeepAgentState
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.types import Command, interrupt

from app.service.ai.middleware.agent_hooks import (
    agent_hooks,
    release_write_resource,
    write_resource_key,
)
from app.service.ai.middleware.interrupt_handler import ConfirmKind, interrupt_handler
from app.service.ai.middleware.run_context import (
    current_subagent_instance,
    enter_subagent_run,
    exit_subagent_run,
    get_run_ctx,
    in_subagent_run,
    set_run_ctx,
    subagent_scope,
)
from app.service.ai.service import trace_collector

logger = logging.getLogger(__name__)


class DehazeAgentState(DeepAgentState):
    """deep agent 状态扩展：在 DeepAgentState 基础上扩展 dehaze 业务上下文与输出契约。

    create_deep_agent 要求 state_schema 为 DeepAgentState 的子类。业务上下文字段
    由 reasoning 层注入 input state，middleware 用 state.get(...) 读取并填充共享
    运行时上下文；输出契约（final_response/usage/stop_reason）由 aafter_agent 写入
    最终 state，供 ainvoke 消费方（EvalRunner / A2A）与 reasoning 层统一读取。
    """

    user_id: NotRequired[int]
    conversation_id: NotRequired[int]
    message_id: NotRequired[int]
    model_id: NotRequired[str]
    stream_session_id: NotRequired[str]
    conversation_prompt: NotRequired[str | None]  # 会话场景提示词，运行时注入，不进图缓存键

    final_response: NotRequired[str]
    usage: NotRequired[dict]
    stop_reason: NotRequired[str]

    step_count: NotRequired[int]
    token_used: NotRequired[int]
    token_budget: NotRequired[int]

    # 多步推理范式（§4）：由 reasoning 层按消息运行时解析注入
    reasoning_mode: NotRequired[str]
    max_steps: NotRequired[int]
    # Plan-and-Execute 计划（Planner/Replanner/计划干预），存图状态以便 resume 续读
    plan: NotRequired[dict]


def _messages_to_dicts(messages: Sequence[BaseMessage] | None) -> list[dict]:
    """将 LangChain 消息列表转为 hooks 兼容的 dict 列表（仅取文本内容）。"""
    if not messages:
        return []
    return [{"role": m.type, "content": m.content} for m in messages if m.content]


def _extract_usage(messages: Sequence[BaseMessage] | None) -> dict:
    """从消息列表中提取最近一条 AI 消息携带的 usage（计费结算透出）。"""
    if not messages:
        return {}
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            usage = (m.response_metadata or {}).get("usage")
            if usage:
                return dict(usage)
    return {}


def _extract_call_meta(messages: Sequence[BaseMessage] | None) -> dict:
    """从消息列表中提取最近一条 AI 消息携带的实际路由归因（降级计费/成本归因透出）。"""
    if not messages:
        return {}
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            meta = (m.response_metadata or {}).get("call_meta")
            if meta:
                return dict(meta)
    return {}


def build_hook_state(
    ctx: dict[str, Any],
    messages: list[dict] | None = None,
    system_prompt: str | None = None,
) -> dict:
    """构造 hooks 兼容的 AgentState dict（计费/护栏钩子的统一入参口径）。

    主循环（awrap_model_call）与范式编排（ParadigmMiddleware 的直连 LLM 调用）
    共用本构造，保证两种路径的步数/Token/预算判定读同一份运行时上下文。
    """
    return {
        "messages": list(messages or []),
        "user_id": ctx.get("user_id"),
        "conversation_id": ctx.get("conversation_id"),
        "message_id": ctx.get("message_id"),
        "model_id": ctx.get("model_id"),
        "system_prompt": system_prompt,
        "stream_session_id": ctx.get("stream_session_id"),
        "is_subagent": in_subagent_run(),
        "billing_context": ctx.get("billing_context"),
        "step_count": ctx.get("step_count", 0),
        "token_used": ctx.get("token_used", 0),
        "token_budget": ctx["token_budget"],
        "max_steps": ctx["max_steps"],
    }


async def run_quota_interrupt(ctx: dict[str, Any], block: dict) -> None:
    """配额中断：持久化中断点（供 resume 恢复）并暂停图（主循环与范式编排共用）。"""
    ctx["stop_reason"] = "quota_exceeded"
    interrupt_data = await _quota_interrupt_data(ctx, block)
    thread_id = f"{ctx.get('conversation_id')}:{ctx.get('message_id')}"
    try:
        await interrupt_handler.save_interrupt(thread_id, "quota", interrupt_data)
    except Exception:
        logger.warning("quota 中断点持久化失败: %s", exc_info=True)
    interrupt(interrupt_data)


async def _quota_interrupt_data(ctx: dict[str, Any], block: dict) -> dict:
    """构造 quota 中断载荷：配额不足时暂停图，用户升级 VIP 后 resume 继续。

    配额信息（已用/限额）最佳努力获取；读取出错不阻断中断（保证用户能升级），
    但须带错误标记，避免伪造"达标"的用量展示。
    """
    tip = block.get("final_response") or "今日或本月 AI 积分配额不足，请升级会员后继续"
    data: dict = {"upgradeTip": tip}
    try:
        from app.database import get_db_session
        from app.service.billing.quota_service import quota_service

        async with get_db_session() as db:
            used_daily, used_monthly = await quota_service.get_used(ctx["user_id"])
            daily_limit, monthly_limit = await quota_service.get_limits(db, ctx["user_id"]) or (
                0,
                0,
            )
        data.update(
            {
                "usedDaily": used_daily,
                "dailyLimit": daily_limit,
                "usedMonthly": used_monthly,
                "monthlyLimit": monthly_limit,
            }
        )
    except Exception:
        logger.error("quota 中断数据组装失败: %s", exc_info=True)
        data["quotaDataError"] = True
    return {
        "type": "quota",
        "stream_session_id": ctx.get("stream_session_id"),
        "data": data,
    }


def _extract_final_response(messages: Sequence[BaseMessage] | None) -> str:
    """从消息列表中提取最终回复文本（最后一条非空 AI 消息内容）。"""
    if not messages:
        return ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            return str(m.content)
    return ""


class DehazeHooksMiddleware(AgentMiddleware):
    """deepagents 中间件：桥接 dehaze AgentHooks 的计费与护栏生命周期。

    Args:
        ctx: 静态配置模板 dict（由 DeepAgentBuilder 创建）。图按版本缓存复用，
            运行时上下文每 run 在 abefore_agent 中经 run_context 独立创建，
            本属性仅作无 run 上下文场景（单测直调）的回退。
        subagent: 本中间件所属的子 Agent 身份 {name, priority}（写冲突仲裁持有者）；
            主 Agent 不传，默认优先级 0（最高）。
    """

    state_schema = DehazeAgentState

    # deepagents 子 Agent 任务工具名：其执行期间整个子图 run 属于子 Agent 口径
    _SUBAGENT_TOOL_NAME = "task"

    def __init__(self, ctx: dict[str, Any], subagent: dict[str, Any] | None = None) -> None:
        self._ctx_template = ctx
        self._subagent = subagent or {"name": "主 Agent", "priority": 0}
        # 子 Agent 标识（主 Agent 为 None）：其模型调用期间经 subagent_scope 绑定到
        # run_context，供用量按运行期归属聚合（不依赖 DB 事后 join）
        self._subagent_code = self._subagent["name"] if subagent else None

    @property
    def ctx(self) -> dict[str, Any]:
        """当前 run 的运行时上下文（abefore_agent 已创建）；无 run 时回退模板"""
        return get_run_ctx() or self._ctx_template

    @property
    def _holder(self) -> dict[str, Any]:
        """写冲突仲裁的持有者身份：Agent 身份 + 本次子 Agent 实例。

        实例 id 由外层 task 工具调用置入 run_context（同一次派发的子图 run 共享），
        同一子 Agent 被并行派发时实例不同，据此判定为冲突而非重入。
        """
        return {**self._subagent, "instance": current_subagent_instance()}

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        # 复用 run() 预置的 run ctx（ensure_run_ctx 在驱动任务中 set，图内全部
        # 节点任务共享同一 dict）；无预置时（单测直调）从模板创建。
        # 主/子 Agent 共享同一 dict：子 Agent 经 task 工具在主 run 上下文内执行，
        # 会话标识天然继承；子 Agent 自身的预算/步数等静态配置并入主 run 口径
        # （按 run 计量而非按 Agent 隔离，防循环观测取全 run 步数）。
        if get_run_ctx() is None:
            set_run_ctx(dict(self._ctx_template))
        is_subagent = in_subagent_run()
        # per-run 字段仅主 run 重置：子 Agent 与主 run 共享同一 dict，
        # 重置会清掉主 run 已累计的步数/预扣阻断等运行状态
        if not is_subagent:
            self.ctx["step_count"] = state.get("step_count", 0)
            self.ctx["token_used"] = state.get("token_used", 0)
            # 范式路由按消息运行时解析出 max_steps（plan_execute/reflexion/direct 各不相同），
            # 图实例按版本缓存复用，故每 run 从 state 覆盖 ctx 的步数上限。
            if state.get("max_steps") is not None:
                self.ctx["max_steps"] = state.get("max_steps")
            self.ctx.pop("stop_reason", None)
            self.ctx.pop("precharge_blocked", None)
            self.ctx.pop("final_response", None)
            self.ctx.pop("multimodal_input_tokens", None)

        # 从 state 提取业务标识与推理起点，填充共享运行时上下文
        # （预置 ctx 已含会话标识，此处按本 run state 幂等覆盖）
        self.ctx["conversation_id"] = state.get("conversation_id") or self.ctx.get(
            "conversation_id"
        )
        self.ctx["user_id"] = state.get("user_id") or self.ctx.get("user_id")
        self.ctx["message_id"] = state.get("message_id") or self.ctx.get("message_id")
        self.ctx["model_id"] = state.get("model_id") or self.ctx.get("model_id")
        self.ctx["stream_session_id"] = state.get("stream_session_id") or self.ctx.get(
            "stream_session_id"
        )
        self.ctx["conversation_prompt"] = state.get("conversation_prompt") or self.ctx.get(
            "conversation_prompt"
        )

        hook_state = self._compat_state(state)
        # 计费预扣：billing_context 写入 hook_state，成功后回填 ctx 贯穿
        result = await agent_hooks.run_hooks("before_agent", hook_state)
        if hook_state.get("billing_context"):
            self.ctx["billing_context"] = hook_state["billing_context"]
        # 前置钩子返回非 None 表示中断，但 abefore_agent 无法直接终止推理，
        # 转为中断标记，由 awrap_model_call 首次调用时短路。
        if result:
            self.ctx["precharge_blocked"] = result
        return None

    def _compat_state(self, state: Any) -> dict:
        """构造 hooks 兼容的 AgentState dict（messages 转 dict 列表）。"""
        return build_hook_state(
            self.ctx, _messages_to_dicts(state.get("messages")), state.get("system_prompt")
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        """子 Agent 模型调用按运行期上下文归属：整个调用在 subagent_scope 内执行，
        采集层据此把本次用量聚合到本子 Agent 的 agentCode（主 Agent 无标识不入子聚合）。"""
        with subagent_scope(self._subagent_code):
            return await self._awrap_model_call(request, handler)

    async def _awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        # 配额不足 → interrupt(type=quota) 暂停图（用户升级 VIP 后 resume 继续），
        # 而非文案短路。预扣阻断（precharge_blocked）与滚动预算不足（before 钩子）
        # 统一收敛于此转为图内中断。
        block = self.ctx.get("precharge_blocked") or {}
        if block.get("stop_reason") == "quota_exceeded":
            await run_quota_interrupt(self.ctx, block)
            # 恢复后预算 hook 重查配额已恢复，清除阻断标记继续推理
            self.ctx.pop("precharge_blocked", None)
            return await handler(request)

        # 步数计数（步数上限/Token 预算判定由 before_model 钩子单一信息源负责）
        self.ctx["step_count"] = self.ctx.get("step_count", 0) + 1
        # 可观测性：同步推理步骤序号，LLM 调用明细经 step_position 关联推理步骤
        trace_collector.set_step_position(self.ctx["step_count"])

        # before_model 钩子框架（步数限制、Token 预算、滚动预算校验等）
        hook_state = self._compat_state(request.state)
        before = await agent_hooks.run_hooks("before_model", hook_state)
        if before:
            # 滚动预算不足：转为 quota 中断（可恢复），而非终止
            if (
                before.get("interrupt", {}).get("type") == "quota"
                or before.get("stop_reason") == "quota_exceeded"
            ):
                await run_quota_interrupt(self.ctx, before)
                self.ctx.pop("precharge_blocked", None)
                return await handler(request)
            # 其余拦截（步数上限/Token 预算）：终止并返回提示
            self.ctx["stop_reason"] = before.get("stop_reason", "blocked")
            return AIMessage(
                content=before.get("final_response", "推理被拦截"),
                response_metadata={"stop_reason": self.ctx["stop_reason"]},
            )

        # 会话场景提示词运行时注入：图缓存键仅含稳定层+Agent 人设，会话层随本次 run
        # 在此并入 system 消息。conversation_prompt 每次 run 恒定，重复并入结果一致。
        conversation_prompt = self.ctx.get("conversation_prompt")
        if conversation_prompt:
            base = request.system_message.content if request.system_message else ""
            composed = f"{base}\n\n{conversation_prompt}" if base else conversation_prompt
            request = request.override(system_message=SystemMessage(content=composed))

        response = await handler(request)
        # after_model 钩子框架 + 累计 token 用量（含工具内多模态视觉读取消耗）
        usage = _extract_usage(response.result)
        await agent_hooks.run_hooks("after_model", {**hook_state, "usage": usage})
        used = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        used += usage.get("completion_tokens") or usage.get("output_tokens") or 0
        used += self.ctx.get("multimodal_input_tokens", 0)
        self.ctx["token_used"] = self.ctx.get("token_used", 0) + used
        return response

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """工具调用写冲突仲裁 + 错误恢复：单工具超时 + 错误分类恢复（§6.3/§6.4）。

        - before_tool 钩子（写冲突仲裁，§5.3.1）：并行子 Agent 对同一资源（文件路径/
          工具目标键）的写入按持有者优先级让行/重试/中断；获取成功则本次调用独占该资源，
          工具结束（含异常）在 finally 释放
        - 用 asyncio.wait_for 施加单工具超时（tool_timeout）；task（子 Agent）工具
          豁免——子 Agent 是多步推理，整体耗时天然超过单工具超时，其内部自有一轮
          步数/Token 护栏
        - 工具异常经 after_tool 钩子分类，按恢复动作处理：
          重试（错误返回 LLM 修正参数）/ 跳过（status=3）/ 失败（status=2）/
          权限不足 → interrupt(type=confirm)
        - 恢复结果（status/error）写入 ToolMessage.additional_kwargs，供 SseEventConverter
          落库 thought 时透出（错误透明告知）
        """
        tool_call = request.tool_call or {}
        tool_call_id = tool_call.get("id", "")
        tool_name = tool_call.get("name", "")

        holder = self._holder
        # 写资源获取：让行/重试/中断冲突全部转为 ToolMessage，不落到工具 handler
        resource = write_resource_key(tool_name, tool_call.get("args"))
        if resource:
            conflict = await self._arbitrate_write(request, resource, holder)
            prompted = False
            while conflict:
                if conflict.get("action") == "interrupt" and not prompted:
                    prompted = True
                    if not await self._do_write_conflict_interrupt(tool_name, resource, conflict):
                        return self._recovery_message(
                            tool_call_id,
                            tool_name,
                            f"用户拒绝覆盖已有写入，本次写入已放弃：{conflict['reason']}",
                            3,
                        )
                    conflict = await self._arbitrate_write(
                        request, resource, holder, overwrite_approved=True
                    )
                    continue
                status = 3 if conflict.get("action") == "skip" else 2
                return self._recovery_message(
                    tool_call_id, tool_name, f"资源写入冲突：{conflict['reason']}", status
                )

        # 子 Agent 任务工具作用域：执行期间标记子 run 口径（计费实报实销/trace 聚合主链），
        # 结束后还原 run 上下文——子图 abefore_agent 会覆盖 ContextVar，
        # 不还原会导致主图后续步骤读到子图 ctx（计费上下文/步数计数错乱）
        is_task_tool = tool_name == self._SUBAGENT_TOOL_NAME
        sub_token = enter_subagent_run(tool_call_id or tool_name) if is_task_tool else None
        parent_ctx = get_run_ctx() if is_task_tool else None
        try:
            if is_task_tool:
                result = await handler(request)
            else:
                result = await asyncio.wait_for(
                    handler(request), timeout=self.ctx.get("tool_timeout") or 60
                )
        except Exception as exc:
            return await self._recover_tool(request, handler, tool_call_id, tool_name, exc)
        finally:
            if resource:
                release_write_resource(self.ctx, resource, holder)
            # sub_token 仅在 is_task_tool 时非空；parent_ctx 为进入 task 工具前捕获的
            # 主 run ctx（其值即当时 _run_ctx，None 时 set_run_ctx(None) 与不设置等价）
            if sub_token is not None:
                exit_subagent_run(sub_token)
            if parent_ctx is not None:
                set_run_ctx(parent_ctx)
        # 成功：原样透传（ToolMessage 默认 status=success，converter 据此落库 status=1）
        return result

    async def _arbitrate_write(
        self, request: Any, resource: str, holder: dict, overwrite_approved: bool = False
    ) -> dict | None:
        """运行 before_tool 钩子（写冲突仲裁），返回 None 表示已获取该写入资源"""
        return await agent_hooks.run_hooks(
            "before_tool",
            {
                **self._compat_state(request.state),
                "run_ctx": self.ctx,
                "resource": resource,
                "holder": holder,
                "retry_max": self.ctx.get("retry_max") or 2,
                "overwrite_approved": overwrite_approved,
            },
        )

    async def _do_write_conflict_interrupt(
        self, tool_name: str, resource: str, conflict: dict
    ) -> bool:
        """写冲突中断：持久化 confirm 中断点并暂停图，由用户裁决是否覆盖既有写入。

        Returns:
            用户是否确认覆盖；拒绝则放弃本次写入（保留既有写入结果）。
        """
        interrupt_data = {
            "type": "confirm",
            "stream_session_id": self.ctx.get("stream_session_id"),
            "data": {
                "confirmKind": ConfirmKind.DANGEROUS_OP,
                "action": "write_conflict",
                "tool": tool_name,
                "resource": resource,
                "previousWriter": conflict.get("previousWriter"),
                "impact": f"{conflict['reason']}；确认后覆盖写入，拒绝则放弃本次写入。",
                "reason": conflict["reason"],
            },
        }
        thread_id = f"{self.ctx.get('conversation_id')}:{self.ctx.get('message_id')}"
        try:
            await interrupt_handler.save_interrupt(thread_id, "confirm", interrupt_data)
        except Exception:
            logger.warning("写冲突中断点保存失败", exc_info=True)
        resume = interrupt(interrupt_data)
        return isinstance(resume, dict) and bool(resume.get("confirmed"))

    async def _recover_tool(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[ToolMessage | Command[Any]]],
        tool_call_id: str,
        tool_name: str,
        exc: Exception,
    ) -> ToolMessage | Command[Any]:
        """工具异常分类恢复：运行 after_tool 钩子，按恢复动作处理并返回 ToolMessage。

        重试计数按 tool_name 而非 tool_call_id：LLM 修正参数重发时会生成新的
        tool_call_id，按 id 计数恒为 0，retry_max 永不生效。
        """
        retry_count = self.ctx.setdefault("tool_retries", {}).get(tool_name) or 0
        hook_state = {
            **self._compat_state(request.state),
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "tool_error": exc,
            "retry_count": retry_count,
            "retry_max": self.ctx.get("retry_max") or 2,
        }
        recovery = await agent_hooks.run_hooks("after_tool", hook_state) or {
            "action": "fail",
            "reason": str(exc),
            "status": 2,
        }

        action = recovery.get("action", "fail")
        reason = recovery.get("reason", str(exc))
        status = int(recovery.get("status", 2))

        if action == "interrupt":
            # 权限确认后重放本次调用：返回失败占位会让 LLM 重试该工具、再次触发
            # 中断形成死循环；本 run 内已授权的同一工具不再二次询问
            authorized = self.ctx.setdefault("authorized_tools", set())
            if tool_name not in authorized:
                resume = await self._do_permission_interrupt(tool_name, reason)
                if not (isinstance(resume, dict) and resume.get("confirmed")):
                    return self._recovery_message(
                        tool_call_id, tool_name, "用户拒绝了该工具的执行", status
                    )
                authorized.add(tool_name)
            return await self._replay_tool(handler, request, tool_call_id, tool_name, status)

        if action == "retry":
            # 参数错误重试：把错误信息作为 ToolMessage 返回 LLM，让其修正参数（上限 retry_max）
            self.ctx.setdefault("tool_retries", {})[tool_name] = retry_count + 1
            return self._recovery_message(
                tool_call_id, tool_name, f"工具参数有误，请修正后重试。{reason}", status
            )

        if action == "skip":
            return self._recovery_message(
                tool_call_id, tool_name, f"该步骤已跳过：{reason}", status
            )

        # fail：记录失败，生成含失败信息的回复
        return self._recovery_message(tool_call_id, tool_name, f"工具调用失败：{reason}", status)

    @staticmethod
    def _recovery_message(
        tool_call_id: str,
        tool_name: str,
        content: str,
        status: int,
    ) -> ToolMessage:
        """构造携带恢复状态（status/error）的 ToolMessage，供 converter 落库透出。"""
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
            additional_kwargs={"_dehaze_status": status, "_dehaze_error": content},
        )

    async def _replay_tool(
        self,
        handler: Callable[[Any], Awaitable[ToolMessage | Command[Any]]],
        request: Any,
        tool_call_id: str,
        tool_name: str,
        status: int,
    ) -> ToolMessage | Command[Any]:
        """重放已获授权的工具调用；再次失败按失败收尾，不再触发中断（防循环）。"""
        try:
            return await handler(request)
        except Exception as exc:
            return self._recovery_message(tool_call_id, tool_name, f"工具执行失败：{exc}", status)

    async def _do_permission_interrupt(self, tool_name: str, reason: str) -> Any:
        """权限不足中断：持久化 confirm 中断点（供 resume 恢复）并暂停图。

        Returns:
            interrupt() 的恢复载荷（resume 传入的 resume_data），用户确认后由
            调用方重放该工具调用。
        """
        interrupt_data = {
            "type": "confirm",
            "stream_session_id": self.ctx.get("stream_session_id"),
            "data": {
                "confirmKind": ConfirmKind.TOOL_PERMISSION,
                "tool": tool_name,
                "reason": reason,
                "detail": "危险操作需用户确认授权后继续",
            },
        }
        thread_id = f"{self.ctx.get('conversation_id')}:{self.ctx.get('message_id')}"
        try:
            await interrupt_handler.save_interrupt(thread_id, "confirm", interrupt_data)
        except Exception:
            logger.warning("权限中断点保存失败", exc_info=True)
        return interrupt(interrupt_data)

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        # 计费实扣结算、记忆提取、会话标题更新
        # 多模态视觉读取的 input_tokens 归集到 usage，随本次推理一并计费
        usage = _extract_usage(state.get("messages"))
        multimodal_tokens = self.ctx.get("multimodal_input_tokens", 0)
        if multimodal_tokens:
            if "input_tokens" in usage:
                usage["input_tokens"] = usage.get("input_tokens", 0) + multimodal_tokens
            else:
                usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + multimodal_tokens
        hook_state = self._compat_state(state)
        hook_state["usage"] = usage
        hook_state["call_meta"] = _extract_call_meta(state.get("messages"))
        await agent_hooks.run_hooks("after_agent", hook_state)

        # 向最终 state 暴露 final_response / usage，供 ainvoke 消费方
        # （EvalRunner / A2A）与 reasoning 层统一读取
        final_response = self.ctx.get("final_response", "")
        if not final_response:
            final_response = _extract_final_response(state.get("messages"))
        self.ctx["final_response"] = final_response
        return {
            "final_response": final_response,
            "usage": usage,
            "stop_reason": self.ctx.get("stop_reason", "stop"),
        }
