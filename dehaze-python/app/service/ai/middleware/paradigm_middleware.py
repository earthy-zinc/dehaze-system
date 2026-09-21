"""ParadigmMiddleware：Plan-and-Execute / Reflexion 范式编排

设计文档 §4。该 middleware 在推理循环入口（abefore_agent）按运行时
state["reasoning_mode"] 分支：

- plan_execute：Planner 分解任务 → 推送 plan 事件 → interrupt 等待用户计划确认/干预
  （resume 透传 plan_edit）→ 依赖拓扑分批并行执行子任务 → 子任务失败走 Replanner
  修订 → 聚合最终答复。reflexion 标注的子任务内嵌 evaluator 迭代（混合架构）。
- reflexion：evaluator 自评 → 低于 reflexion_threshold 时 self_reflection 生成改进
  策略并写入反思记忆（source=reflection）→ 下一轮注入 → 超 max_iterations_reflexion
  后接受当前最佳。
- react/direct：不介入，走主图 ReAct 或 direct 直连。

图按 (agent_id, version_no) 缓存复用，middleware 恒定装载，仅按运行时的
reasoning_mode 决定是否介入，因此不破坏图缓存。

范式内 LLM 调用直连 DehazeChatModel，但同样经 before_model/after_model 钩子
（步数/Token/滚动预算护栏与主循环同一判定源），并累计 step_count/token_used；
多次调用的 usage 汇总挂到最终消息的 response_metadata，随 jump_to=end 交给
after_agent 完成计费结算/trace/记忆等收尾钩子。

范式相关配置从 snapshot config 读取：max_parallel、max_iterations_reflexion、
reflexion_threshold、reflexion_expected_format。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from app.infrastructure.llm.client.dehaze_chat_model import DehazeChatModel
from app.service.ai.middleware.agent_hooks import agent_hooks
from app.service.ai.middleware.dehaze_hooks_middleware import (
    build_hook_state,
    run_quota_interrupt,
)
from app.service.ai.middleware.interrupt_handler import interrupt_handler
from app.service.ai.middleware.run_context import get_run_ctx
from app.service.ai.paradigms import plan_execute
from app.service.ai.paradigms import reflexion as reflexion_mod
from app.service.ai.strategies.quota_recall import quota_recall

logger = logging.getLogger(__name__)

# 计划确认中断类型（resume 端点据此渲染计划卡片）
PLAN_APPROVE_TYPE = "plan_approve"

# Replanner 修订轮次上限：修订会重算批次，无上限时反复失败会无限重规划
_MAX_REPLAN_ROUNDS = 5


class _ParadigmBlocked(Exception):
    """护栏拦截（步数上限/Token 预算）：终止范式编排并按拦截文案收尾。"""

    def __init__(self, block: dict) -> None:
        super().__init__(block.get("final_response") or "推理被拦截")
        self.block = block


class _ParadigmUsage:
    """范式内 LLM 调用的用量汇总与护栏拦截标记（每次编排一个实例）。"""

    def __init__(self) -> None:
        self.totals = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
        # 实际路由归因取最后一次调用（降级计费/成本归因口径与消息提取一致）
        self.call_meta: dict = {}
        self.block: dict | None = None

    def add(self, usage: dict) -> int:
        """汇总单次调用 usage，返回本次 token 量（input+output，供 Token 预算累计）。"""
        input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        self.totals["input_tokens"] += input_tokens
        self.totals["output_tokens"] += output_tokens
        self.totals["cached_input_tokens"] += int(usage.get("cached_input_tokens") or 0)
        return input_tokens + output_tokens

    def raise_if_blocked(self) -> None:
        """补抛护栏拦截：build_plan/replan 内部吞掉模型异常，需按标记判定。"""
        if self.block:
            raise _ParadigmBlocked(self.block)


class ParadigmMiddleware(AgentMiddleware):
    """多步推理范式编排中间件。"""

    def __init__(
        self,
        model: DehazeChatModel,
        config: dict[str, Any],
        ctx: dict[str, Any],
        *,
        save_memory: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        # 图按版本缓存复用，ctx 为静态模板；运行时上下文由 DehazeHooksMiddleware
        # 每 run 经 run_context 创建（计划状态随 run 隔离），无 run 时回退模板
        self._ctx_template = ctx
        # 反思记忆落库钩子（默认走 memory_extraction 的保存能力，可注入 mock）
        self._save_memory = save_memory

    @property
    def ctx(self) -> dict[str, Any]:
        return get_run_ctx() or self._ctx_template

    # ── AgentMiddleware 钩子 ────────────────────────────

    # 允许从 before_agent 跳到 end（范式编排完成后直接收尾）
    @hook_config(can_jump_to=["end"])
    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        mode = state.get("reasoning_mode") or "react"
        if mode not in ("plan_execute", "reflexion"):
            return None
        usage = _ParadigmUsage()
        try:
            if mode == "plan_execute":
                return await self._run_plan_execute(state, runtime, usage)
            return await self._run_reflexion(state, runtime, usage)
        except _ParadigmBlocked as e:
            stop_reason = e.block.get("stop_reason") or "blocked"
            logger.warning("范式推理被护栏拦截 [mode=%s stop_reason=%s]", mode, stop_reason)
            return self._finish(
                state,
                None,
                e.block.get("final_response") or "推理被拦截",
                usage,
                stop_reason,
            )

    # ── 编排入口 ────────────────────────────────────────

    async def _run_plan_execute(
        self, state: Any, runtime: Any, usage: _ParadigmUsage
    ) -> dict[str, Any] | None:
        """Plan-and-Execute 主流程：计划 → 确认/干预 → 分批并行执行 → 聚合。"""
        model_call = self._make_model_call(state, usage)

        # StreamWriter 是同步函数（Callable[[Any], None]），这里包装为 async 以统一
        # _PlanExecutor 的 await emit(...) 契约（直接 await 同步返回值会 TypeError）
        async def emit(event: dict[str, Any]) -> None:
            runtime.stream_writer(event)

        # 计划优先取图 state（checkpoint 持久化）；plan_approve 中断后本节点整体重跑、
        # 中断前的 state 写入随中断丢弃，故再从中断点记录恢复用户已确认的计划
        plan = state.get("plan") or await self._restore_pending_plan(state)
        if plan is None:
            plan = await plan_execute.build_plan(self._last_user_content(state), model_call)
            usage.raise_if_blocked()
            await emit({"type": "plan", "data": {"plan": plan, "phase": "plan"}})
            await self._emit_thought(runtime, tool="planner", observation=_plan_summary(plan))

        # 计划确认/干预：status=executing 表示本轮已确认过；否则经 plan_approve 中断
        # 等待用户 resume（透传 plan_edit），resume 时 interrupt() 返回确认载荷。
        if plan.get("status") != "executing":
            await self._await_plan_approval(state, plan)
            await emit({"type": "plan", "data": {"plan": plan, "phase": "approved"}})

        executor = _PlanExecutor(
            model_call=model_call,
            max_parallel=int(self.config.get("max_parallel") or 1),
            emit=emit,
            reflexion_cfg=self.config,
            ctx=self.ctx,
            usage=usage,
        )
        plan = await executor.run(plan)
        await emit({"type": "plan", "data": {"plan": plan, "phase": "done"}})

        return self._finish(
            state, plan, _compose_plan_answer(plan, self._last_user_content(state)), usage
        )

    async def _run_reflexion(
        self, state: Any, runtime: Any, usage: _ParadigmUsage
    ) -> dict[str, Any] | None:
        """Reflexion 主流程：actor 执行 → evaluator 自评 → reflection → 迭代。"""
        model_call = self._make_model_call(state, usage)
        requirement = self._last_user_content(state)
        max_iterations = int(self.config.get("max_iterations_reflexion") or 1)
        threshold = float(self.config.get("reflexion_threshold") or 0.8)
        expected = self.config.get("reflexion_expected_format")

        loop = reflexion_mod.reflexion_loop(
            run_actor=model_call,
            evaluate=lambda req, out: reflexion_mod.evaluate_output(
                req, out, model_call, expected=expected
            ),
            reflect=lambda req, out, fb: reflexion_mod.reflect_failure(req, out, fb, model_call),
            max_iterations=max_iterations,
            threshold=threshold,
        )
        best, rounds = await loop(requirement, [])
        usage.raise_if_blocked()

        # 最后一轮反思写入记忆（source=reflection），供同任务再执行时注入
        reflection = rounds[-1].get("strategy")
        if reflection:
            await self._save_reflection_memory(
                state,
                requirement,
                {
                    "root_cause": rounds[-1].get("feedback", ""),
                    "strategy": reflection,
                },
            )
        await self._emit_thought(
            runtime,
            tool="evaluator",
            observation=(
                f"自评最高分 {max((r['score'] for r in rounds), default=0):.2f}，"
                f"共 {len(rounds)} 轮"
            ),
        )
        return self._finish(state, None, best, usage)

    # ── 内部工具 ────────────────────────────────────────

    def _finish(
        self,
        state: Any,
        plan: dict[str, Any] | None,
        response: str,
        usage: _ParadigmUsage,
        stop_reason: str = "stop",
    ) -> dict[str, Any]:
        update: dict[str, Any] = {
            # 范式编排完成即收尾：跳到 after_agent（计费结算/trace/记忆钩子在此执行），
            # 不再进入 model 节点重复调用
            "jump_to": "end",
            # usage 汇总挂到最终消息，供 after_agent 的结算/trace 钩子提取
            "messages": [
                AIMessage(
                    content=response,
                    response_metadata={"usage": dict(usage.totals), "call_meta": usage.call_meta},
                )
            ],
            "final_response": response,
            "stop_reason": stop_reason,
        }
        if plan is not None:
            update["plan"] = plan
        return update

    def _make_model_call(
        self, state: Any, usage: _ParadigmUsage
    ) -> Callable[[list[dict], str], Awaitable[str]]:
        """构造 (messages, system_prompt) -> str 的模型调用闭包。"""

        async def _call(messages: list[dict], system_prompt: str) -> str:
            hook_state = build_hook_state(self.ctx, messages, system_prompt)
            block = await agent_hooks.run_hooks("before_model", hook_state)
            if block:
                if (
                    block.get("interrupt", {}).get("type") == "quota"
                    or block.get("stop_reason") == "quota_exceeded"
                ):
                    await run_quota_interrupt(self.ctx, block)
                    self.ctx.pop("precharge_blocked", None)
                else:
                    usage.block = block
                    raise _ParadigmBlocked(block)
            self.ctx["step_count"] = self.ctx.get("step_count", 0) + 1
            lm_messages = [SystemMessage(content=system_prompt)]
            lm_messages += [
                HumanMessage(content=m["content"]) for m in messages if m.get("content")
            ]
            result = await self.model.ainvoke(lm_messages)
            meta = result.response_metadata or {}
            llm_usage = meta.get("usage") or {}
            used = usage.add(llm_usage)
            usage.call_meta = dict(meta.get("call_meta") or {})
            hook_state["usage"] = llm_usage
            await agent_hooks.run_hooks("after_model", hook_state)
            self.ctx["token_used"] = self.ctx.get("token_used", 0) + used
            return str(result.content or "")

        return _call

    @staticmethod
    def _last_user_content(state: Any) -> str:
        for m in reversed(state.get("messages") or []):
            if getattr(m, "type", "") == "human" and m.content:
                return str(m.content)
        return ""

    async def _restore_pending_plan(self, state: Any) -> dict[str, Any] | None:
        """从 plan_approve 中断点恢复计划。

        中断后本节点整体重跑、中断前的 state 写入随中断丢弃，故用户已确认的计划
        以中断点记录为准（避免 resume 时重建计划、用户确认与干预丢失）。
        """
        thread_id = f"{state.get('conversation_id')}:{state.get('message_id')}"
        try:
            record = await interrupt_handler.get_interrupt(thread_id)
        except Exception:
            logger.warning("计划中断点读取失败，按新计划处理", exc_info=True)
            return None
        if (record or {}).get("type") != PLAN_APPROVE_TYPE:
            return None
        record_data = (record or {}).get("data") or {}
        return (record_data.get("data") or {}).get("plan") or None

    async def _await_plan_approval(self, state: Any, plan: dict[str, Any]) -> None:
        """计划确认中断：暂停图等待用户 resume（透传 plan_edit 干预）。

        resume 时 interrupt() 返回 resume_data，据此合并计划干预并执行。
        """
        thread_id = f"{state.get('conversation_id')}:{state.get('message_id')}"
        interrupt_data = {
            "type": PLAN_APPROVE_TYPE,
            "stream_session_id": state.get("stream_session_id"),
            "data": {"plan": plan},
        }
        try:
            await interrupt_handler.save_interrupt(thread_id, PLAN_APPROVE_TYPE, interrupt_data)
        except Exception:
            # 中断点仅为 resume 恢复依据，Redis 抖动不得令本轮推理失败
            logger.warning("计划确认中断点持久化失败: %s", thread_id, exc_info=True)
        resume_data = interrupt(interrupt_data) or {}
        if resume_data.get("plan_edit"):
            try:
                plan_execute.apply_plan_edit(plan, resume_data["plan_edit"])
            except ValueError as e:
                # 干预窗口校验失败：记录并忽略干预，按原计划执行
                logger.warning("计划干预被拒绝: %s", e)
        plan["status"] = "executing"

    async def _save_reflection_memory(self, state: Any, requirement: str, reflection: dict) -> None:
        """把反思结果写入 sys_ai_memory（source=reflection），供检索注入。"""
        if self._save_memory is None:
            return
        try:
            memory = reflexion_mod.build_reflection_memory(
                user_id=state.get("user_id"),
                conversation_id=state.get("conversation_id"),
                model_id=state.get("model_id"),
                requirement=requirement,
                reflection=reflection,
                skill=self.config.get("reflexion_skill"),
            )
            await self._save_memory(memory)
        except Exception:
            logger.warning("反思记忆写入失败", exc_info=True)

    async def _emit_thought(self, runtime: Any, tool: str, observation: str, **extra: Any) -> None:
        """经 custom 事件通道推送 thought（由 SseEventConverter 公开接口落库+推 SSE）。"""
        runtime.stream_writer(
            {
                "type": "thought",
                "data": {"tool": tool, "observation": observation, **extra},
            }
        )


class _PlanExecutor:
    """计划执行器：按依赖拓扑分批并行执行子任务，失败走 Replanner。

    子任务以自然语言指令委派给 LLM（DehazeChatModel），批内并行执行、
    max_parallel 限流；paradigm=reflexion 的子任务执行后走 evaluator 迭代。

    配额召回（§7.2）：派发前按批 precharge_batch 预留，不足则整批降级跳过；
    批结束 settle_batch 退回预留；并行执行中 check_and_recall 召回未启动子任务
    （记 skipped），正在执行的等当前步完成后自然收尾，已完成的保留。
    """

    def __init__(
        self,
        model_call: Callable[[list[dict], str], Awaitable[str]],
        max_parallel: int,
        emit: Callable[[dict], Any],
        reflexion_cfg: dict[str, Any],
        ctx: dict[str, Any],
        usage: _ParadigmUsage,
    ) -> None:
        self.model_call = model_call
        self.max_parallel = max(1, max_parallel)
        self.emit = emit
        self.reflexion_cfg = reflexion_cfg
        self.ctx = ctx
        self.usage = usage

    async def run(self, plan: dict[str, Any]) -> dict[str, Any]:
        # Replanner 会替换/新增子任务，批次须按最新计划重算，否则修订出的任务
        # 不在本轮批次内、永远停在 pending 且被最终聚合遗漏
        for _ in range(_MAX_REPLAN_ROUNDS):
            batches = _pending_batches(plan)
            if not batches:
                break
            revised = False
            for batch in batches:
                revised = await self._run_one_batch(plan, batch)
                if revised:
                    break
            if not revised:
                break
        # 修订轮次耗尽后仍未执行的任务按失败收尾，避免停在 pending 被聚合遗漏
        for task in plan.get("tasks") or []:
            if task.get("status") == "pending":
                task["status"] = "failed"
                task["result"] = "修订次数耗尽，未执行"
        plan["status"] = "done"
        return plan

    async def _run_one_batch(self, plan: dict[str, Any], batch_ids: list[str]) -> bool:
        """执行单个依赖批，返回是否触发 Replanner 修订。"""
        tasks_map = {t["id"]: t for t in plan.get("tasks") or []}
        tasks = [tasks_map[tid] for tid in batch_ids if tid in tasks_map]
        if not tasks:
            return False
        reserved = await quota_recall.precharge_batch(self.ctx, tasks)
        if reserved is None:
            for task in tasks:
                task.update(status="failed", result="配额不足，本批未执行")
            return False

        usage_before = dict(self.usage.totals)
        try:
            results = await self._run_batch(plan, batch_ids)
        finally:
            # 批内异常/取消同样释放预留：预留是真实配额扣减（叠加在主预扣之上），
            # 泄漏会一直占用用户配额直到日/月额度重置
            await quota_recall.settle_batch(
                self.ctx, reserved, _delta(usage_before, self.usage.totals)
            )
        for task_id, ok, text in results:
            task = tasks_map.get(task_id)
            if not task:
                continue
            task["status"] = "done" if ok else "failed"
            task["result"] = text
            if not ok:
                logger.warning("子任务 %s 失败: %s", task_id, text)

        # 批内存在失败 → Replanner 修订受影响部分后重排计划
        failed_ids = [task_id for task_id, ok, _ in results if not ok]
        if not failed_ids:
            return False
        plan = await plan_execute.replan(plan, failed_ids, self.model_call)
        self.usage.raise_if_blocked()
        await self.emit({"type": "plan", "data": {"plan": plan, "phase": "revised"}})
        return True

    async def _run_batch(
        self, plan: dict[str, Any], batch_ids: list[str]
    ) -> list[tuple[str, bool, str]]:
        """在单个依赖批内并行执行子任务，支持配额召回未启动任务。"""
        results: dict[str, tuple[bool, str]] = {}
        lock = asyncio.Lock()
        running: set[str] = set()
        pending = list(batch_ids)

        async def worker() -> None:
            while True:
                task_id = None
                recalled = 0
                async with lock:
                    # 配额召回为纯 ctx 判定（无 IO），在临界区内完成；召回提示的
                    # SSE 推送放到临界区外，避免持锁 await 把批内并行退化为串行
                    for tid in quota_recall.check_and_recall(self.ctx, pending):
                        pending.remove(tid)
                        results[tid] = (False, "配额不足未执行")
                        recalled += 1
                    if pending:
                        task_id = pending.pop(0)
                        running.add(task_id)
                if recalled:
                    await quota_recall.notify_partial_skipped(self.ctx, recalled)
                if task_id is None:
                    return
                # 执行当前子任务（进行中的等当前步完成，不中断）
                try:
                    ok, text = await self._execute_task(plan, task_id)
                except asyncio.CancelledError:
                    async with lock:
                        running.discard(task_id)
                        results[task_id] = (False, "执行被取消")
                    return
                async with lock:
                    running.discard(task_id)
                    results[task_id] = (ok, text)

        workers = [
            asyncio.create_task(worker()) for _ in range(min(self.max_parallel, len(batch_ids)))
        ]
        await asyncio.gather(*workers)
        return [(tid, *results[tid]) for tid in batch_ids if tid in results]

    async def _execute_task(self, plan: dict, task_id: str) -> tuple[bool, str]:
        task = next(t for t in plan.get("tasks") or [] if t["id"] == task_id)
        context = _plan_context(plan, task_id)
        if task.get("paradigm") == "reflexion":
            return await self._reflexion_task(task, context)
        try:
            text = await self.model_call(
                [{"role": "user", "content": f"{task['description']}\n\n上下文：{context}"}],
                _SUBTASK_SYSTEM_PROMPT,
            )
            return bool(text.strip()), text.strip()
        except _ParadigmBlocked:
            raise
        except Exception as e:
            # 单子任务异常不应炸掉整批：降级为失败结果交给 Replanner；
            # 但"执行异常"从结果文本回灌会丢失异常栈与归因，必须留痕
            logger.error("范式子任务执行异常 [task_id=%s]: %s", task_id, e, exc_info=True)
            return False, f"执行异常: {e}"

    async def _reflexion_task(self, task: dict, context: str) -> tuple[bool, str]:
        loop = reflexion_mod.reflexion_loop(
            run_actor=lambda msgs, prompt: self.model_call(
                [{"role": "user", "content": f"{prompt}\n\n上下文：{context}"}],
                _SUBTASK_SYSTEM_PROMPT,
            ),
            evaluate=lambda req, out: reflexion_mod.evaluate_output(
                req,
                out,
                self.model_call,
                expected=self.reflexion_cfg.get("reflexion_expected_format"),
            ),
            reflect=lambda req, out, fb: reflexion_mod.reflect_failure(
                req, out, fb, self.model_call
            ),
            max_iterations=int(self.reflexion_cfg.get("max_iterations_reflexion") or 1),
            threshold=float(self.reflexion_cfg.get("reflexion_threshold") or 0.8),
        )
        best, _rounds = await loop(task["description"], [])
        return bool(best.strip()), best.strip()


_SUBTASK_SYSTEM_PROMPT = (
    "你是子任务执行者。根据子任务说明与上下文，完成该子任务并直接返回结果内容，"
    "不要复述任务要求，不要输出多余解释。"
)


def _delta(before: dict, after: dict) -> dict:
    """取两次 usage 快照的增量（本批实际消耗）。"""
    return {key: after.get(key, 0) - before.get(key, 0) for key in after}


def _pending_batches(plan: dict[str, Any]) -> list[list[str]]:
    """按最新计划计算待执行任务的并行批次（已完成/失败任务的依赖视为已满足）。"""
    tasks = [t for t in plan.get("tasks") or [] if t.get("status") == "pending"]
    pending_ids = {t["id"] for t in tasks}
    return plan_execute.compute_batches(
        [
            {**t, "depends_on": [d for d in t.get("depends_on") or [] if d in pending_ids]}
            for t in tasks
        ]
    )


def _plan_context(plan: dict[str, Any], task_id: str) -> str:
    """构造当前任务的可参考上下文（已完成任务的结论）。"""
    done = [
        f"[{t['id']}] {t['description']}: {t.get('result')}"
        for t in plan.get("tasks") or []
        if t.get("status") == "done" and t.get("result")
    ]
    return "\n".join(done) or "（尚无已完成子任务）"


def _plan_summary(plan: dict[str, Any]) -> str:
    tasks = plan.get("tasks") or []
    return f"分解任务为 {len(tasks)} 个子任务: " + "、".join(t["id"] for t in tasks)


def _compose_plan_answer(plan: dict[str, Any], user_task: str = "") -> str:
    """聚合所有子任务结果为最终答复。"""
    done = [t for t in plan.get("tasks") or [] if t.get("status") == "done" and t.get("result")]
    failed = [t for t in plan.get("tasks") or [] if t.get("status") == "failed"]
    parts = []
    if done:
        parts.append(
            "已完成各子任务：\n" + "\n".join(f"- {t['description']}: {t['result']}" for t in done)
        )
    if failed:
        parts.append("以下子任务未能完成：" + "、".join(t["id"] for t in failed))
    return "\n\n".join(parts) or f"任务「{user_task}」计划执行完成。"
