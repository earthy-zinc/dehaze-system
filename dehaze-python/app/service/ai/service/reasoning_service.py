"""推理编排服务（ReasoningService）

基于 deepagents 引擎（run/resume/stop 幂等、后台任务、SSE、
落库 sys_ai_message / sys_ai_agent_thought、计费结算）：

- run：读会话锚定的 Agent 版本快照 → 组装 deepagents 图（deep agent / Team）→
  astream(version="v2", subgraphs=True) 流式推理 → SseEventConverter 推 SSE →
  从最终 state 提取 final_response/usage → 落库结算
- resume：处理用户确认 → Command(resume=...) 恢复中断的推理
- stop：停止推理

图按 (agent_id, version_no) 缓存（Agent 发布/回滚不影响进行中会话，锚定版本不可变）。
"""

import asyncio
import json
import logging
import traceback
from typing import Any

from langgraph.types import Command

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db_session
from app.infrastructure.sse.sse_emitter_manager import sse_emitter_manager
from app.repository.ai_conversation_repository import ai_conversation_repository
from app.repository.ai_message_repository import ai_message_repository
from app.repository.ai_trace_repository import ai_trace_repository
from app.infrastructure.cache.checkpoint_manager import checkpoint_manager
from app.service.ai.builders.context_manager import context_manager
from app.service.ai.service.conversation_search_service import sync_conversation_to_es
from app.service.ai.service import trace_collector
from app.service.ai.service.credits_service import calculate_credits
from app.service.ai.builders.deep_agent_builder import DeepAgentBuilder
from app.service.ai.middleware.interrupt_handler import interrupt_handler
from app.infrastructure.sse.sse_event_converter import SseEventConverter
from app.service.ai.service.step_summarizer import schedule_step_summaries
from app.service.ai.service.suggestion_service import suggestion_service
from app.service.ai.service.summary_service import summary_service
from app.service.ai.builders.team_builder import TeamBuilder
from app.service.billing.billing_service import billing_service

logger = logging.getLogger(__name__)

# 异步后台任务引用，防止被垃圾回收
_pending_tasks: set[asyncio.Task] = set()

# 默认 Agent 编码（与后端实现 §2.1 一致，未指定 Agent 时的兜底）
DEFAULT_AGENT_CODE = "default"


def _schedule_conversation_sync(conv_id: int) -> None:
    """异步同步会话到 ES 全文索引（消息落库后触发，不阻塞主流程）"""

    async def _run() -> None:
        try:
            await sync_conversation_to_es(conv_id)
        except Exception:
            logger.warning("Conversation ES sync failed", exc_info=True)

    task = asyncio.create_task(_run())
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)


class ReasoningService:
    """Agent 推理编排服务（单例）"""

    def __init__(self) -> None:
        # 图缓存：{(agent_id, version_no): CompiledStateGraph}
        self._graphs: dict[tuple[int, int], Any] = {}

    @staticmethod
    def _thread_id(conv_id: int, msg_id: int) -> str:
        return f"{conv_id}:{msg_id}"

    async def _load_agent_anchor(self, db, conv) -> tuple[int, int]:
        """解析会话锚定的 (agent_id, agent_version)。

        会话无 agent_code 时用默认 Agent；版本锚定缺失时取当前已发布版本。
        """
        from app.repository.ai_agent_repository import ai_agent_repository

        agent_code = conv.agent_code or DEFAULT_AGENT_CODE
        agent = await ai_agent_repository.get_by_code(db, agent_code)
        if not agent or agent.deleted:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "Agent 不存在")
        return agent.id, conv.agent_version

    async def _load_snapshot(self, db, redis, agent_id: int, version_no: int | None) -> dict:
        """按锚定版本加载已发布快照（不含 checkpointer，供范式路由等只读用途）。"""
        from app.service.ai_agent_service import agent_service

        return await agent_service.get_published_snapshot(db, redis, agent_id, version_no)

    async def _build_graph(
        self, db, redis, agent_id: int, version_no: int | None, model_id: str | None = None
    ) -> Any:
        """按锚定版本组装推理图（deep agent 或 Team），图实例按版本+模型缓存。

        会话模型（model_id）覆盖 Agent 快照默认模型（三级合并"会话覆盖"原则），
        参与图缓存键；同一 Agent 不同会话模型各自建图。
        """
        key = (agent_id, version_no or 0, model_id or "")
        if key in self._graphs:
            return self._graphs[key]

        from app.service.ai_agent_service import agent_service

        snapshot = await agent_service.get_published_snapshot(db, redis, agent_id, version_no)
        if model_id:
            # 会话模型覆盖 Agent 默认模型：snapshot 仅"模型"字段被覆盖，
            # 其余（提示词/工具/护栏）仍随 Agent 版本快照保持不可变。
            snapshot["model_id"] = model_id
        checkpointer = checkpoint_manager.get_checkpointer()
        if snapshot.get("is_team"):
            rels = snapshot.get("subagents") or []
            member_snapshots = [
                await agent_service.get_published_snapshot(db, redis, rel.get("agent_id"))
                for rel in rels
                if not rel.get("endpoint_id")
            ]
            remote_members = [rel for rel in rels if rel.get("endpoint_id")]
            graph = await TeamBuilder.build_team(
                db,
                redis,
                snapshot,
                member_snapshots,
                remote_members=remote_members,
                checkpointer=checkpointer,
            )
        else:
            graph = await DeepAgentBuilder.build_from_snapshot(
                db, redis, snapshot, checkpointer=checkpointer
            )
        self._graphs[key] = graph
        return graph

    @staticmethod
    def _state_result(state: Any) -> dict:
        """从最终图 state 提取 {final_response, stop_reason, usage}。

        langgraph 的 aget_state 返回对象必含 values（AgentState 基类字段）；若缺失
        说明推理异常中断，不得伪装成成功态消费，直接显式抛错暴露。
        """
        values = state.values
        if not values:
            logger.error("推理未产出 state.values，图可能异常中断，禁止按成功态消费")
            raise RuntimeError("推理未产出有效 state.values，图可能异常中断")
        usage = values.get("usage") or {}
        return {
            "final_response": values.get("final_response", ""),
            "stop_reason": values.get("stop_reason", "stop"),
            "usage": {
                "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens") or 0,
                "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens") or 0,
                "cached_input_tokens": usage.get("cached_input_tokens", 0),
            },
        }

    async def _finalize_message(
        self, msg_id: int, result: dict, model_id: str, used_memory_ids: list[int] | None = None
    ) -> int:
        """推理完成后更新 assistant 消息内容、token 统计、注入记忆可见性与积分，返回积分"""
        usage = result.get("usage") or {}
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
        cached_input_tokens = usage.get("cached_input_tokens", 0)
        async with get_db_session() as db:
            msg = await ai_message_repository.get_by_id(db, msg_id)
            if msg:
                msg.content = result.get("final_response", "")
                msg.input_tokens = input_tokens
                msg.output_tokens = output_tokens
                msg.cached_input_tokens = cached_input_tokens
                if used_memory_ids:
                    msg.used_memory_ids = used_memory_ids
                credits = await calculate_credits(
                    db, model_id, input_tokens, output_tokens, cached_input_tokens
                )
                msg.credits = credits
                msg.status = 2
                await db.flush()
                return credits
        return 0

    async def _push_end(self, stream_session_id: str, result: dict, credits: int = 0) -> None:
        """推送 message.end 事件"""
        usage = result.get("usage") or {}
        await sse_emitter_manager.send_event(
            stream_session_id,
            "message.end",
            {
                "stopReason": result.get("stop_reason", "stop"),
                "usage": {
                    "inputTokens": usage.get("input_tokens", 0),
                    "outputTokens": usage.get("output_tokens", 0),
                    "cachedInputTokens": usage.get("cached_input_tokens", 0),
                    "credits": credits,
                },
            },
        )

    @staticmethod
    def _trigger_suggestions(
        conv_id: int,
        msg_id: int,
        result: dict,
        user_id: int,
        stream_session_id: str,
    ) -> None:
        """message.end 推送后异步生成类似问题推荐，不阻塞主回复完成。"""
        reply_content = result.get("final_response", "") or ""
        if result.get("stop_reason") in ("canceled", "error"):
            return

        async def _run() -> None:
            try:
                await suggestion_service.generate(
                    conversation_id=conv_id,
                    message_id=msg_id,
                    reply_content=reply_content,
                    user_id=user_id,
                    stream_session_id=stream_session_id,
                )
            except Exception:
                logger.warning("类似问题推荐触发异常", exc_info=True)

        task = asyncio.create_task(_run())
        _pending_tasks.add(task)
        task.add_done_callback(_pending_tasks.discard)

    async def _fail(self, msg_id: int, stream_session_id: str, error: Exception) -> None:
        """推理失败：更新消息状态为失败并推送 error 事件"""
        try:
            async with get_db_session() as db:
                await ai_message_repository.update_status(db, msg_id, 3, str(error))
        except Exception:
            pass
        await sse_emitter_manager.send_event(
            stream_session_id, "error", {"code": "A0600", "message": str(error)}
        )
        # error 后补 message.end 收尾，保证客户端总能走到统一完成处理
        await sse_emitter_manager.send_event(
            stream_session_id,
            "message.end",
            {
                "stopReason": "error",
                "usage": {
                    "inputTokens": 0,
                    "outputTokens": 0,
                    "cachedInputTokens": 0,
                    "credits": 0,
                },
            },
        )

    async def run(
        self,
        conv_id: int,
        user_id: int,
        msg_id: int,
        model_id: str,
        stream_session_id: str,
    ) -> dict:
        """启动 Agent 推理，返回 {final_response, stop_reason, usage}

        上下文（messages/system_prompt/记忆注入）在本方法内由 build_context 一次性组装，
        调用方无需预热；单次发送仅执行一次 build_context，避免记忆 touch 副作用翻倍。
        """
        from app.dependencies.redis import get_redis_client

        injected_list: list[dict] = []
        async with get_db_session() as db:
            conv = await ai_conversation_repository.get_by_id_and_user(db, conv_id, user_id)
            if conv:
                await summary_service.maybe_compress(db, conv, model_id)
                messages, system_prompt, injected_list = await context_manager.build_context(
                    db, conv, model_id
                )
            # 可观测性：开启过程链采集并记录上下文构成快照（§2.2）
            trace_collector.start(
                conversation_id=conv_id,
                message_id=msg_id,
                user_id=user_id,
                agent_code=getattr(conv, "agent_code", None),
                model_id=model_id,
            )
            if conv:
                trace_collector.current().record_context(
                    system_prompt=system_prompt,
                    messages=messages,
                    injected_memories=injected_list,
                    summary=getattr(conv, "summary", None),
                )
            agent_id, version_no = await self._load_agent_anchor(db, conv)
            redis = await get_redis_client()
            snapshot = await self._load_snapshot(db, redis, agent_id, version_no)

            # 范式路由：按快照 reasoning_mode（auto 时经复杂度评估）解析实际范式与步数上限。
            # direct 不构建完整 deepagents 图，走单次 LLM 直连（无工具），真实获得性能收益。
            reasoning_mode, max_steps = await DeepAgentBuilder.resolve_reasoning_mode(
                snapshot, messages, model_id
            )
            if reasoning_mode != "direct":
                graph = await self._build_graph(db, redis, agent_id, version_no, model_id)

        if reasoning_mode == "direct":
            return await self._run_direct(
                conv_id,
                user_id,
                msg_id,
                model_id,
                stream_session_id,
                messages,
                system_prompt,
            )

        used_memory_ids = [item.get("memory_id") for item in injected_list if item.get("memory_id")]
        initial_state = {
            "messages": messages,
            "user_id": user_id,
            "conversation_id": conv_id,
            "message_id": msg_id,
            "model_id": model_id,
            "system_prompt": system_prompt,
            "stream_session_id": stream_session_id,
            # 会话场景提示词经运行时注入，不进入图缓存键
            "conversation_prompt": getattr(conv, "system_prompt", None),
            # 注入记忆可见性，写入图状态以便 resume 从 checkpoint 续读
            "used_memory_ids": used_memory_ids,
            "step_count": 0,
            "token_used": 0,
            # 范式路由结果写入图状态：reasoning_mode 供 ParadigmMiddleware 分支，
            # max_steps 供 DehazeHooksMiddleware 每 run 覆盖 ctx 步数上限。
            "reasoning_mode": reasoning_mode,
            "max_steps": max_steps,
        }
        config = {"configurable": {"thread_id": self._thread_id(conv_id, msg_id)}}
        converter = SseEventConverter({**initial_state})
        try:
            async for event in graph.astream(
                initial_state,
                config=config,
                stream_mode=["messages", "updates", "custom"],
                version="v2",
                subgraphs=True,
            ):
                await converter.handle(event)
            final_state = await graph.aget_state(config)
        except Exception as e:
            logger.error("Agent 推理失败: %s", e, exc_info=True)
            await trace_collector.finalize_unsettled(
                status=trace_collector.TRACE_STATUS_FAILED, error_type=trace_collector.error_type_of(e)
            )
            await self._fail(msg_id, stream_session_id, e)
            raise

        # 推理中断挂起（图暂停待确认）：释放会话并发锁让渡给 resume 续流。
        # 挂起期间不再产生并发写，锁可安全让渡（release 幂等，
        # 即使 create_stream 已释放也无副作用）。
        interrupt_data = await interrupt_handler.get_interrupt(self._thread_id(conv_id, msg_id))
        if interrupt_data:
            await sse_emitter_manager.release_lock(conv_id)
            # 本轮以中断态收尾（resume 由新请求 trace_id 记独立过程链），
            # error_type 记录中断类型（quota/confirm/async_wait），供审计侧直接检索
            await trace_collector.finalize_unsettled(
                status=trace_collector.TRACE_STATUS_INTERRUPTED,
                error_type=(interrupt_data or {}).get("type"),
            )

        result = self._state_result(final_state)
        # async_wait 挂起：回复尚未完成，前端凭 interrupt.data.task_id 轮询消息接口等
        # 最终态。此处跳过落库置 2/步骤摘要/建议触发，仅推 message.end 作为本轮流结束
        # 信号；消息保持创建时的"生成中"状态，待 resume 完成后再置 2 写最终 content。
        is_async_wait_suspend = (interrupt_data or {}).get("type") == "async_wait"
        if is_async_wait_suspend:
            credits = 0
        else:
            credits = await self._finalize_message(msg_id, result, model_id, used_memory_ids)
            # 异步生成步骤摘要（两级展示一级），不阻塞主回复
            schedule_step_summaries(conv_id, msg_id, model_id)
            self._trigger_suggestions(conv_id, msg_id, result, user_id, stream_session_id)
        await converter.finish()  # 文本/思考内容块收尾（message.end 前）
        await self._push_end(stream_session_id, result, credits)
        _schedule_conversation_sync(conv_id)
        return result

    async def _run_direct(
        self,
        conv_id: int,
        user_id: int,
        msg_id: int,
        model_id: str,
        stream_session_id: str,
        messages: list[dict],
        system_prompt: str | None,
    ) -> dict:
        """direct 范式：单次 LLM 直连（无工具），跳过 deepagents 图，最小化推理开销。

        与常规路径共用 converter/finalize/push_end，仅在消息组装与流式来源上走
        轻量路径；返回 {final_response, stop_reason, usage} 契约与 run 一致。

        计费与图路径对齐（图路径经 before_agent/after_agent 钩子完成，此处直接调用）：
        推理前 pre_charge 预扣，欠费/配额/余额不足时以中断文案作为回复阻断推理；
        推理后 settle 实扣结算。direct 无检查点可恢复，配额阻断为终态提示
        （用户升级后重新提问），不提供图路径的 quota 中断挂起/resume 语义。
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        from app.infrastructure.llm.client.dehaze_chat_model import DehazeChatModel

        converter = SseEventConverter(
            {
                "conversation_id": conv_id,
                "message_id": msg_id,
                "model_id": model_id,
                "stream_session_id": stream_session_id,
            }
        )
        # 可观测性：direct 范式无图钩子，采集与结算在本方法内完成
        trace_collector.start(
            conversation_id=conv_id,
            message_id=msg_id,
            user_id=user_id,
            agent_code=None,
            model_id=model_id,
        )
        trace_collector.current().record_context(
            system_prompt=system_prompt, messages=messages, injected_memories=[], summary=None
        )
        lm_messages: list = []
        if system_prompt:
            lm_messages.append(SystemMessage(content=system_prompt))
        # 完整对话历史按序传入（user/assistant/system），direct 范式不能只传最后一句，
        # 否则模型看不到上文，无法回答"我说了什么/复述我的回答"等依赖历史的问题
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if not content:
                continue
            if role == "user":
                lm_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lm_messages.append(AIMessage(content=content))
            elif role == "system":
                lm_messages.append(SystemMessage(content=content))
        last_human = next(
            (
                m.get("content")
                for m in reversed(messages)
                if m.get("role") == "user" and m.get("content")
            ),
            "",
        )

        # 上下文预算保护：完整历史可能超出模型窗口，按预算从最早消息裁剪
        # （保留 index0 的 system_prompt 与最近对话；count_tokens 为本地字符估算）
        from app.infrastructure.llm.call.llm_client import llm_client
        from app.repository.ai_model_repository import ai_model_repository

        async with get_db_session() as db:
            model_entity = await ai_model_repository.get_by_model_id(db, model_id)
        budget = None
        if model_entity:
            output_reserve = model_entity.max_output_tokens or 1024
            budget = max(model_entity.max_context_tokens - output_reserve - 200, 512)
        if budget:
            total = 0
            for m in lm_messages:
                total += await llm_client.count_tokens(m.content)
            while total > budget and len(lm_messages) > 2:
                # 优先裁剪最早的非 system 对话消息（保留 system_prompt/摘要/记忆与最近对话）
                idx = next(
                    (i for i, m in enumerate(lm_messages) if not isinstance(m, SystemMessage)),
                    None,
                )
                if idx is None or idx == 0:
                    break
                dropped = lm_messages.pop(idx)
                total -= await llm_client.count_tokens(dropped.content)

        async with get_db_session() as db:
            billing_ctx = await billing_service.pre_charge(
                db, user_id, conv_id, msg_id, last_human, model_id
            )
        if "billing_id" not in billing_ctx:
            # 欠费/配额/余额不足：不调用 LLM，中断文案作为本条回复
            result = {
                "final_response": billing_ctx["final_response"],
                "stop_reason": billing_ctx["stop_reason"],
                "usage": {},
            }
            await trace_collector.finalize_unsettled(
                status=trace_collector.TRACE_STATUS_FAILED,
                error_type=(billing_ctx.get("stop_reason") or "precharge_blocked")[:32],
            )
            credits = await self._finalize_message(msg_id, result, model_id)
            await converter.finish()
            await self._push_end(stream_session_id, result, credits)
            return result

        model = DehazeChatModel(model=model_id)
        full_text = ""
        usage: dict = {}
        try:
            # langchain_core 1.5+ 的 BaseChatModel.astream() 直接产出 AIMessageChunk
            # （不再包装为 ChatGenerationChunk），converter.handle 以 [AIMessageChunk, meta]
            # 消费，直接透传 chunk 即可；token 用量经 _astream 聚合到 model._last_usage，
            # 流结束后统一读取，不依赖逐 chunk 的 response_metadata。
            async for chunk in model.astream(lm_messages):
                if chunk.content:
                    full_text += chunk.content
                    await converter.handle({"type": "messages", "data": [chunk, {}]})
            usage = dict(model._last_usage or {})
        except Exception as e:
            logger.error("direct 推理失败: %s", e, exc_info=True)
            await trace_collector.finalize_unsettled(
                status=trace_collector.TRACE_STATUS_FAILED,
                error_type=trace_collector.error_type_of(e),
                error_detail={"message": str(e)[:500], "stack": traceback.format_exc()[:4000]},
            )
            await self._fail(msg_id, stream_session_id, e)
            raise

        await converter.finish()
        result = {
            "final_response": full_text,
            "stop_reason": "stop",
            # OpenAI 兼容流下发 prompt/completion_tokens 键名，统一为内部
            # input/output 键（与图路径 _state_result 一致），供 _push_end/settle 消费
            "usage": {
                "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens") or 0,
                "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens") or 0,
                "cached_input_tokens": usage.get("cached_input_tokens", 0),
            },
        }
        credits = await self._finalize_message(msg_id, result, model_id)
        async with get_db_session() as db:
            await billing_service.settle(
                db, user_id, conv_id, msg_id, model_id, None, usage
            )
        await trace_collector.finalize_success(usage=result.get("usage"), step_count=1)
        await self._push_end(stream_session_id, result, credits)
        _schedule_conversation_sync(conv_id)
        return result

    async def resume(self, conv_id: int, user_id: int, msg_id: int, resume_data: dict) -> dict:
        """恢复中断的推理，返回 {final_response, stop_reason, usage}"""
        thread_id = self._thread_id(conv_id, msg_id)
        interrupt = await interrupt_handler.get_interrupt(thread_id)
        if not interrupt:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "未找到中断点，无法恢复")
        # 可观测性：resume 续流以新请求 trace_id 独立成链，此处开启采集器；
        # 中断恢复决策摘要先取原值（quota 分支会覆盖 resume_data）
        decision_summary = json.dumps(resume_data, ensure_ascii=False, default=str)[:500]
        trace_collector.start(
            conversation_id=conv_id,
            message_id=msg_id,
            user_id=user_id,
            agent_code=None,
            model_id=None,
        )
        stream_session_id = (interrupt.get("data") or {}).get("stream_session_id", "")
        config = {"configurable": {"thread_id": thread_id}}
        from app.dependencies.redis import get_redis_client

        try:
            # 按中断类型构造恢复载荷：
            # - confirm：用户确认/拒绝算法推荐，resume_data 为 {confirmed, algorithmId}
            # - async_wait：异步任务结果，resume_data 为 {async_task: summary}，注入工具中断点
            # - quota：用户升级 VIP 后直接从中断点继续（预算 hook 重查配额），resume=True
            if interrupt.get("type") == "confirm":
                from app.service.ai.service.algorithm_recommend_service import handle_user_confirmation

                await handle_user_confirmation(
                    conv_id,
                    msg_id,
                    user_id,
                    resume_data.get("confirmed", False),
                    resume_data.get("algorithmId"),
                )
            elif interrupt.get("type") == "quota":
                resume_data = True
            async with get_db_session() as db:
                conv = await ai_conversation_repository.get_by_id_and_user(db, conv_id, user_id)
                agent_id, version_no = await self._load_agent_anchor(db, conv)
                redis = await get_redis_client()
                # 会话模型覆盖 Agent 默认模型（三级合并"会话覆盖"原则），
                # resume 与 run 使用同一模型建图，保证 checkpoint 续读一致。
                msg0 = await ai_message_repository.get_by_id(db, msg_id)
                resume_model = msg0.model if msg0 else ""
                graph = await self._build_graph(db, redis, agent_id, version_no, resume_model)
                interrupted_trace = await ai_trace_repository.get_latest_by_message_and_status(
                    db, msg_id, trace_collector.TRACE_STATUS_INTERRUPTED
                )
            # 可观测性：补全采集器归属（start 时会话/模型尚未加载），记录中断恢复决策
            collector = trace_collector.current()
            collector.agent_code = getattr(conv, "agent_code", None)
            collector.model_id = resume_model or None
            collector.record_event(
                event="resume",
                interrupt_type=interrupt.get("type"),
                decision=decision_summary,
                from_trace_id=interrupted_trace.trace_id if interrupted_trace else None,
            )
            # 恢复推理并推送 SSE 事件（thought/进度/文本增量续流）
            converter = SseEventConverter(
                {
                    "stream_session_id": stream_session_id,
                    "message_id": msg_id,
                    "conversation_id": conv_id,
                }
            )
            async for event in graph.astream(
                Command(resume=resume_data),
                config=config,
                stream_mode=["messages", "updates", "custom"],
                version="v2",
                subgraphs=True,
            ):
                await converter.handle(event)
            await converter.finish()
            final_state = await graph.aget_state(config)
        except Exception as e:
            logger.error("Agent 推理恢复失败: %s", e, exc_info=True)
            await trace_collector.finalize_unsettled(
                status=trace_collector.TRACE_STATUS_FAILED,
                error_type=trace_collector.error_type_of(e),
                error_detail={"message": str(e)[:500], "stack": traceback.format_exc()[:4000]},
            )
            await self._fail(msg_id, stream_session_id, e)
            raise
        await interrupt_handler.clear_interrupt(thread_id)
        async with get_db_session() as db:
            msg = await ai_message_repository.get_by_id(db, msg_id)
            model_id = msg.model if msg else ""
        result = self._state_result(final_state)
        # 注入记忆可见性：resume 未重走 build_context，从 checkpoint state 续读 used_memory_ids
        # （_state_result 已校验 state.values 非空）
        used_memory_ids = final_state.values.get("used_memory_ids") or None
        credits = await self._finalize_message(msg_id, result, model_id, used_memory_ids)
        # 可观测性：resume 无 after_agent 兜底时的成功收尾（幂等，钩子已结算则跳过）
        await trace_collector.finalize_success(
            usage=result.get("usage"), step_count=final_state.values.get("step_count", 0)
        )
        if stream_session_id:
            await self._push_end(stream_session_id, result, credits)
        _schedule_conversation_sync(conv_id)
        return result

    async def stop(self, conv_id: int, msg_id: int, stream_session_id: str) -> None:
        """停止推理（用户主动中断）：结束流、更新消息状态、清除中断点"""
        await sse_emitter_manager.stop_stream(stream_session_id)
        async with get_db_session() as db:
            await ai_message_repository.update_status(db, msg_id, 4)
        await interrupt_handler.clear_interrupt(self._thread_id(conv_id, msg_id))


reasoning_service = ReasoningService()
