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
reasoning_mode 决定是否介入，因此不破坏图缓存。所有 LLM 调用经 DehazeChatModel
（自管理 db/redis）直接完成，不计入主图 step/token 钩子，由本 middleware 通过
custom 事件（plan/thought）统一汇报。

范式相关配置从 snapshot config 读取：max_parallel、max_iterations_reflexion、
reflexion_threshold、reflexion_expected_format。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command, interrupt

from app.service.ai.dehaze_chat_model import DehazeChatModel
from app.service.ai.interrupt_handler import interrupt_handler
from app.service.ai.paradigms import plan_execute
from app.service.ai.paradigms import reflexion as reflexion_mod
from app.service.ai.quota_recall import quota_recall

logger = logging.getLogger(__name__)

# 计划确认中断类型（resume 端点据此渲染计划卡片）
PLAN_APPROVE_TYPE = "plan_approve"


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
        self.ctx = ctx
        # 反思记忆落库钩子（默认走 memory_extraction 的保存能力，可注入 mock）
        self._save_memory = save_memory

    # ── AgentMiddleware 钩子 ────────────────────────────

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        mode = state.get("reasoning_mode") or "react"
        if mode == "plan_execute":
            return await self._run_plan_execute(state, runtime)
        if mode == "reflexion":
            return await self._run_reflexion(state, runtime)
        return None

    # 允许从 before_agent 跳到 end（范式编排完成后直接收尾）
    abefore_agent.__can_jump_to__ = ["end"]

    # ── 编排入口 ────────────────────────────────────────

    async def _run_plan_execute(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Plan-and-Execute 主流程：计划 → 确认/干预 → 分批并行执行 → 聚合。"""
        model_call = self._make_model_call(state)

        # StreamWriter 是同步函数（Callable[[Any], None]），这里包装为 async 以统一
        # _PlanExecutor 的 await emit(...) 契约（直接 await 同步返回值会 TypeError）
        async def emit(event: dict[str, Any]) -> None:
            runtime.stream_writer(event)

        msg_id = state.get("message_id")

        # 计划经 ctx（按 message_id 隔离）在 plan_approve 中断的 resume 续流间持久化：
        # 首次进入生成；resume 时复用已生成计划并合并 plan_edit 干预。
        plan_key = f"_plan_{msg_id}"
        approved_key = f"_plan_approved_{msg_id}"
        plan = self.ctx.get(plan_key)
        if plan is None:
            user_task = self._last_user_content(state)
            plan = await plan_execute.build_plan(user_task, model_call)
            self.ctx[plan_key] = plan
            await emit({"type": "plan", "data": {"plan": plan, "phase": "plan"}})
            await self._emit_thought(runtime, tool="planner", observation=_plan_summary(plan))

        # 计划确认/干预：首次进入经 plan_approve 中断等待用户 resume（透传 plan_edit）；
        # resume 续流时 interrupt() 返回确认载荷并合并干预。
        if not self.ctx.get(approved_key):
            await self._await_plan_approval(state, plan)
            self.ctx[approved_key] = True
            await emit({"type": "plan", "data": {"plan": plan, "phase": "approved"}})

        # 混合范式：reflexion 子任务并行执行（含 evaluator 迭代）
        executor = _PlanExecutor(
            model_call=model_call,
            max_parallel=int(self.config.get("max_parallel") or 1),
            emit=emit,
            reflexion_cfg=self.config,
            ctx=self.ctx,
        )
        plan = await executor.run(plan)
        await emit({"type": "plan", "data": {"plan": plan, "phase": "done"}})
        # 清理本消息的计划状态，避免残留影响后续 run
        self.ctx.pop(plan_key, None)
        self.ctx.pop(approved_key, None)

        final_response = _compose_plan_answer(plan, user_task=self._last_user_content(state))
        return self._finish(state, plan, final_response)

    async def _run_reflexion(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Reflexion 主流程：actor 执行 → evaluator 自评 → reflection → 迭代。"""
        model_call = self._make_model_call(state)
        requirement = self._last_user_content(state)
        max_iterations = int(self.config.get("max_iterations_reflexion") or 1)
        threshold = float(self.config.get("reflexion_threshold") or 0.8)
        expected = self.config.get("reflexion_expected_format")

        loop = reflexion_mod.reflexion_loop(
            run_actor=lambda messages, prompt: self._actor_call(messages, prompt),
            evaluate=lambda req, out: reflexion_mod.evaluate_output(
                req, out, model_call, expected=expected
            ),
            reflect=lambda req, out, fb: reflexion_mod.reflect_failure(req, out, fb, model_call),
            max_iterations=max_iterations,
            threshold=threshold,
        )
        best, rounds = await loop(requirement, [])

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
        return self._finish(state, None, best)

    # ── 内部工具 ────────────────────────────────────────

    def _finish(self, state: Any, plan: dict[str, Any] | None, response: str) -> dict[str, Any]:
        update: dict[str, Any] = {
            "messages": [AIMessage(content=response)],
            "final_response": response,
            "stop_reason": "stop",
        }
        if plan is not None:
            update["plan"] = plan
        return Command(goto="end", update=update)

    def _make_model_call(self, state: Any) -> Callable[[list[dict], str], Awaitable[str]]:
        """构造 (messages, system_prompt) -> str 的模型调用闭包。"""

        async def _call(messages: list[dict], system_prompt: str) -> str:
            lm_messages = [SystemMessage(content=system_prompt)]
            lm_messages += [
                HumanMessage(content=m["content"]) for m in messages if m.get("content")
            ]
            result = await self.model.ainvoke(lm_messages)
            return str(result.content or "")

        return _call

    async def _actor_call(self, messages: list[dict], prompt: str) -> str:
        return await self._make_model_call({})(messages, prompt)

    @staticmethod
    def _last_user_content(state: Any) -> str:
        for m in reversed(state.get("messages") or []):
            if getattr(m, "type", "") == "human" and m.content:
                return str(m.content)
        return ""

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
        await interrupt_handler.save_interrupt(thread_id, PLAN_APPROVE_TYPE, interrupt_data)
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

    配额召回（§7.2）：派发前按批 precharge_batch 预扣，失败则整批降级跳过；
    并行执行中 check_and_recall 召回未启动子任务（记 skipped），正在执行的等当前步
    完成后自然收尾，已完成的保留。
    """

    def __init__(
        self,
        model_call: Callable[[list[dict], str], Awaitable[str]],
        max_parallel: int,
        emit: Callable[[dict], Any],
        reflexion_cfg: dict[str, Any],
        ctx: dict[str, Any],
    ) -> None:
        self.model_call = model_call
        self.max_parallel = max(1, max_parallel)
        self.emit = emit
        self.reflexion_cfg = reflexion_cfg
        self.ctx = ctx

    async def run(self, plan: dict[str, Any]) -> dict[str, Any]:
        tasks_map = {t["id"]: t for t in plan.get("tasks") or []}
        for batch in plan_execute.compute_batches(plan.get("tasks") or []):
            batch_ids = [tid for tid in batch if tid in tasks_map]
            if not batch_ids:
                continue
            # 派发前批量预扣；预扣失败则整批降级跳过（配额不足）
            if not await quota_recall.precharge_batch(self.ctx, len(batch_ids)):
                for tid in batch_ids:
                    tasks_map[tid].update(status="failed", result="配额不足，本批未执行")
                continue
            results = await self._run_batch(batch_ids, tasks_map)
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
            if failed_ids:
                plan = await plan_execute.replan(plan, failed_ids, self.model_call)
                await self.emit({"type": "plan", "data": {"plan": plan, "phase": "revised"}})
                tasks_map = {t["id"]: t for t in plan.get("tasks") or []}
        plan["status"] = "done"
        return plan

    async def _run_batch(
        self, batch_ids: list[str], tasks_map: dict[str, dict]
    ) -> list[tuple[str, bool, str]]:
        """在单个依赖批内并行执行子任务，支持配额召回未启动任务。"""
        results: dict[str, tuple[bool, str]] = {}
        lock = asyncio.Lock()
        running: set[str] = set()
        pending = list(batch_ids)

        async def worker() -> None:
            while True:
                async with lock:
                    # 执行中配额不足：召回全部未启动子任务
                    if not pending:
                        return
                    recalled = await quota_recall.check_and_recall(
                        self.ctx, list(running), list(pending)
                    )
                    for tid in recalled:
                        pending.remove(tid)
                        results[tid] = (False, "配额不足未执行")
                    if not pending:
                        return
                    tid = pending.pop(0)
                    running.add(tid)
                # 执行当前子任务（进行中的等当前步完成，不中断）
                try:
                    ok, text = await self._execute_task(tasks_map[tid], tasks_map)
                except asyncio.CancelledError:
                    async with lock:
                        running.discard(tid)
                        results[tid] = (False, "执行被取消")
                    return
                async with lock:
                    running.discard(tid)
                    results[tid] = (ok, text)

        workers = [
            asyncio.create_task(worker()) for _ in range(min(self.max_parallel, len(batch_ids)))
        ]
        await asyncio.gather(*workers)
        return [(tid, *results[tid]) for tid in batch_ids if tid in results]

    async def _execute_task(self, task: dict, plan: dict) -> tuple[bool, str]:
        context = _plan_context(plan, task["id"])
        if task.get("paradigm") == "reflexion":
            return await self._reflexion_task(task, context)
        try:
            text = await self.model_call(
                [{"role": "user", "content": f"{task['description']}\n\n上下文：{context}"}],
                _SUBTASK_SYSTEM_PROMPT,
            )
            return bool(text.strip()), text.strip()
        except Exception as e:
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
        best, rounds = await loop(task["description"], [])
        return bool(best.strip()), best.strip()


# 子任务执行系统提示词
_SUBTASK_SYSTEM_PROMPT = (
    "你是子任务执行者。根据子任务说明与上下文，完成该子任务并直接返回结果内容，"
    "不要复述任务要求，不要输出多余解释。"
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
