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
import logging
from typing import Any

from langgraph.types import Command

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db_session
from app.infrastructure.sse.sse_emitter_manager import sse_emitter_manager
from app.repository.ai_conversation_repository import ai_conversation_repository
from app.repository.ai_message_repository import ai_message_repository
from app.service.ai.checkpoint_manager import checkpoint_manager
from app.service.ai.context_manager import context_manager
from app.service.ai.conversation_search_service import sync_conversation_to_es
from app.service.ai.credits_service import calculate_credits
from app.service.ai.deep_agent_builder import DeepAgentBuilder
from app.service.ai.interrupt_handler import interrupt_handler
from app.service.ai.sse_event_converter import SseEventConverter
from app.service.ai.step_summarizer import schedule_step_summaries
from app.service.ai.suggestion_service import suggestion_service
from app.service.ai.summary_service import summary_service
from app.service.ai.team_builder import TeamBuilder

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
        from app.service.ai_agent_service import AgentService

        return await AgentService().get_published_snapshot(db, redis, agent_id, version_no)

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

        from app.service.ai_agent_service import AgentService

        snapshot = await AgentService().get_published_snapshot(db, redis, agent_id, version_no)
        if model_id:
            # 会话模型覆盖 Agent 默认模型：snapshot 仅"模型"字段被覆盖，
            # 其余（提示词/工具/护栏）仍随 Agent 版本快照保持不可变。
            snapshot["model_id"] = model_id
        checkpointer = checkpoint_manager.get_checkpointer()
        if snapshot.get("is_team"):
            rels = snapshot.get("subagents") or []
            member_snapshots = [
                await AgentService().get_published_snapshot(db, redis, rel.get("agent_id"))
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
        """从最终图 state 提取 {final_response, stop_reason, usage}。"""
        values = getattr(state, "values", None) or (state if isinstance(state, dict) else {})
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
            await self._fail(msg_id, stream_session_id, e)
            raise

        # 推理中断挂起（图暂停待确认）：释放会话并发锁让渡给 resume 续流。
        # 挂起期间不再产生并发写，锁可安全让渡（release 幂等，
        # 即使 create_stream 已释放也无副作用）。
        interrupt_data = await interrupt_handler.get_interrupt(self._thread_id(conv_id, msg_id))
        if interrupt_data:
            await sse_emitter_manager.release_lock(conv_id)

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
            schedule_step_summaries(msg_id, model_id)
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
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        from app.service.ai.dehaze_chat_model import DehazeChatModel

        converter = SseEventConverter(
            {
                "conversation_id": conv_id,
                "message_id": msg_id,
                "model_id": model_id,
                "stream_session_id": stream_session_id,
            }
        )
        lm_messages: list = []
        if system_prompt:
            lm_messages.append(SystemMessage(content=system_prompt))
        last_human = next(
            (
                m.get("content")
                for m in reversed(messages)
                if m.get("role") == "user" and m.get("content")
            ),
            "",
        )
        lm_messages.append(HumanMessage(content=last_human))

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
            await self._fail(msg_id, stream_session_id, e)
            raise

        await converter.finish()
        result = {
            "final_response": full_text,
            "stop_reason": "stop",
            "usage": usage,
        }
        credits = await self._finalize_message(msg_id, result, model_id)
        await self._push_end(stream_session_id, result, credits)
        _schedule_conversation_sync(conv_id)
        return result

    async def resume(self, conv_id: int, user_id: int, msg_id: int, resume_data: dict) -> dict:
        """恢复中断的推理，返回 {final_response, stop_reason, usage}"""
        thread_id = self._thread_id(conv_id, msg_id)
        interrupt = await interrupt_handler.get_interrupt(thread_id)
        if not interrupt:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "未找到中断点，无法恢复")
        # 按中断类型构造恢复载荷：
        # - confirm：用户确认/拒绝算法推荐，resume_data 为 {confirmed, algorithmId}
        # - async_wait：异步任务结果，resume_data 为 {async_task: summary}，注入工具中断点
        # - quota：用户升级 VIP 后直接从中断点继续（预算 hook 重查配额），resume=True
        if interrupt.get("type") == "confirm":
            from app.service.ai.algorithm_recommend_service import handle_user_confirmation

            await handle_user_confirmation(
                conv_id,
                msg_id,
                user_id,
                resume_data.get("confirmed", False),
                resume_data.get("algorithmId"),
            )
        elif interrupt.get("type") == "quota":
            resume_data = True
        stream_session_id = (interrupt.get("data") or {}).get("stream_session_id", "")
        config = {"configurable": {"thread_id": thread_id}}
        from app.dependencies.redis import get_redis_client

        try:
            async with get_db_session() as db:
                conv = await ai_conversation_repository.get_by_id_and_user(db, conv_id, user_id)
                agent_id, version_no = await self._load_agent_anchor(db, conv)
                redis = await get_redis_client()
                # 会话模型覆盖 Agent 默认模型（三级合并"会话覆盖"原则），
                # resume 与 run 使用同一模型建图，保证 checkpoint 续读一致。
                msg0 = await ai_message_repository.get_by_id(db, msg_id)
                resume_model = msg0.model if msg0 else ""
                graph = await self._build_graph(db, redis, agent_id, version_no, resume_model)
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
            await self._fail(msg_id, stream_session_id, e)
            raise
        await interrupt_handler.clear_interrupt(thread_id)
        async with get_db_session() as db:
            msg = await ai_message_repository.get_by_id(db, msg_id)
            model_id = msg.model if msg else ""
        result = self._state_result(final_state)
        # 注入记忆可见性：resume 未重走 build_context，从 checkpoint state 续读 used_memory_ids
        state_values = getattr(final_state, "values", None) or (
            final_state if isinstance(final_state, dict) else {}
        )
        used_memory_ids = state_values.get("used_memory_ids") or None
        credits = await self._finalize_message(msg_id, result, model_id, used_memory_ids)
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
