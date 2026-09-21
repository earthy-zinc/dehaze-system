"""上下文组装器（ContextManager）

每次推理前组装发送给 LLM 的消息列表：
system_prompt → summary（system 消息）→ 长期记忆 → token 预算内的对话历史原文
（含产物引用行）。超出预算的更早消息经 SummaryService 增量摘要兜底（§4.1），
不会静默丢失。
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.llm.call.llm_client import llm_client
from app.repository.ai_message_repository import ai_message_repository
from app.repository.ai_model_repository import ai_model_repository
from app.service.ai.service import trace_collector
from app.service.ai.service.memory_injection import inject_memories
from app.service.ai.strategies.prompt_composer import compose_system_prompt
from app.service.ai_artifact_service import ai_artifact_service

logger = logging.getLogger(__name__)

# 链上消息单次查询上限（仅作查询边界；实际窗口按 token 预算动态裁剪）
_CONTEXT_FETCH_LIMIT = 200
# 组装余量：工具定义等不参与窗口预算的开销，预留 token 空间
_CONTEXT_MARGIN_TOKENS = 2000
# token 预算下限（模型窗口极小时仍保留最小对话空间）
_CONTEXT_BUDGET_FLOOR = 1024
# 模型未注册时的保守窗口兜底：不裁剪会把整条链原文送入，必然超窗
_FALLBACK_CONTEXT_TOKENS = 32768
_FALLBACK_OUTPUT_TOKENS = 4096
# 产物引用行中 summary 字段的最大展示长度
_ARTIFACT_SUMMARY_MAX_LEN = 200
# 工具调用结果折叠进上下文的单条截断长度（防超大工具输出撑爆上下文）
_TOOL_RESULT_MAX_LEN = 1000


class ContextManager:
    """上下文组装器（单例）"""

    async def build_context(
        self,
        db: AsyncSession,
        conv,
        model_id: str,
    ) -> tuple[list[dict], str | None, list[dict]]:
        """组装发送给 LLM 的上下文消息列表，返回 (messages, system_prompt, injected_list)。

        - system_prompt：完整组装结果（稳定层 + Agent 人设 + 会话场景提示词），
          与图运行时实际发送给 LLM 的系统消息保持一致。
        - injected_list：本次注入的长期记忆清单（[{memory_id, memory_type, content, source}]），
          供推理层落库 used_memory_ids 做注入可见性。
        """
        agent_snapshot = await self._load_agent_snapshot(db, conv)
        system_prompt = compose_system_prompt(agent_snapshot, conv)
        messages = await self._load_candidate_messages(db, conv)
        # 长期记忆注入（对话消息之前；摘要层位于记忆之后，见会话与消息 §6.1 组装顺序）：
        # inject_memories 返回 (system_block_text, injected_list)，
        # system 补充块作为 system 消息注入。
        last_user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user" and msg["content"]:
                last_user_msg = msg["content"]
                break
        system_block, injected_list = await inject_memories(db, conv.user_id, last_user_msg)
        # 记忆块以独立 system 消息注入、不参与历史裁剪，须先从窗口预算中扣除
        reserved = await llm_client.count_tokens(system_block) if system_block else 0
        await self._trim_to_budget(db, conv, model_id, system_prompt, messages, reserved)
        if system_block:
            messages.insert(0, {"role": "system", "content": system_block})
        if conv.summary:
            insert_at = 1 if system_block else 0
            messages.insert(
                insert_at, {"role": "system", "content": f"之前的对话摘要：{conv.summary}"}
            )
        return messages, system_prompt, injected_list

    @staticmethod
    async def _load_agent_snapshot(db: AsyncSession, conv) -> dict | None:
        """按会话锚定的 Agent 解析其已发布版本快照（用于组装完整 system_prompt）。"""
        if not conv or not getattr(conv, "agent_code", None):
            return None
        from app.dependencies.redis import get_redis_client
        from app.repository.ai_agent_repository import ai_agent_repository
        from app.service.ai_agent_service import agent_service

        agent = await ai_agent_repository.get_by_code(db, conv.agent_code)
        if not agent or agent.deleted:
            return None
        try:
            redis = await get_redis_client()
            return await agent_service.get_published_snapshot(
                db, redis, agent.id, conv.agent_version
            )
        except Exception as e:  # 快照加载失败不阻断上下文组装
            logger.warning("加载 Agent 快照失败(agent=%s): %s", conv.agent_code, e)
            return None

    @staticmethod
    async def _window_budget(
        db: AsyncSession, model_id: str, system_prompt: str | None, summary: str | None
    ) -> int:
        """对话历史 token 预算：模型窗口扣除输出预留/系统提示/摘要/组装余量。

        模型未注册时按保守窗口推算：此时放弃裁剪会把整条链原文送入，必然超窗。
        """
        model = await ai_model_repository.get_by_model_id(db, model_id)
        if not model:
            logger.warning(
                "模型 %s 未在模型注册表登记，上下文预算按保守窗口 %s 推算",
                model_id,
                _FALLBACK_CONTEXT_TOKENS,
            )
            context_tokens, output_tokens = _FALLBACK_CONTEXT_TOKENS, _FALLBACK_OUTPUT_TOKENS
        else:
            context_tokens = model.max_context_tokens
            output_tokens = model.max_output_tokens or 1024
        system_tokens = await llm_client.count_tokens(system_prompt or "")
        summary_tokens = await llm_client.count_tokens(summary or "")
        return max(
            _CONTEXT_BUDGET_FLOOR,
            context_tokens
            - output_tokens
            - system_tokens
            - summary_tokens
            - _CONTEXT_MARGIN_TOKENS,
        )

    @staticmethod
    async def _load_candidate_messages(db: AsyncSession, conv) -> list[dict]:
        """加载分支链上的候选对话历史（沿 current_branch_message_id 回溯）。

        过滤已软删行，只取 user/assistant 且 content 非空，按时间正序。
        工具调用结果（role=tool）折叠为文本追加到前一条 assistant 消息，
        使后续轮次模型能看到历史工具调用结果；保留 message_id 以支撑产物引用关联。
        产物引用行在此追加（裁剪前），使其 token 计入窗口预算。
        """
        msgs = await ai_message_repository.get_chain_by_id(
            db,
            conv.id,
            conv.current_branch_message_id,
            limit=_CONTEXT_FETCH_LIMIT,
        )
        result: list[dict] = []
        for m in msgs:
            if m.role == "tool" and m.content:
                # 工具结果折叠进前一条 assistant 消息（无对应 assistant 时忽略，
                # 避免孤立 tool 文本破坏 user/assistant 交替结构）
                if result and result[-1]["role"] == "assistant":
                    tool_text = m.content
                    if len(tool_text) > _TOOL_RESULT_MAX_LEN:
                        tool_text = tool_text[:_TOOL_RESULT_MAX_LEN] + "…"
                    result[-1]["content"] = f"{result[-1]['content']}\n[工具调用结果] {tool_text}"
                continue
            if m.role in ("user", "assistant") and m.content:
                result.append({"id": m.id, "role": m.role, "content": m.content})
        # 关联产物引用行（按 message_id 查询，仅在对应消息后追加引用，绝不注入全文/URL）
        await ContextManager._attach_artifact_refs(db, result)
        return result

    @staticmethod
    async def _trim_to_budget(
        db: AsyncSession,
        conv,
        model_id: str,
        system_prompt: str | None,
        messages: list[dict],
        reserved_tokens: int,
    ) -> None:
        """按窗口预算裁剪最早消息（截断事件写过程链，由 SummaryService 摘要兜底）。

        reserved_tokens 为窗口内但不参与裁剪的内容（长期记忆块）已占用的 token。
        """
        budget = await ContextManager._window_budget(db, model_id, system_prompt, conv.summary)
        budget -= reserved_tokens
        token_list = [await llm_client.count_tokens(m["content"]) for m in messages]
        total = sum(token_list)
        cut = 0
        # 从最早消息起裁剪至预算内（至少保留最后一条，保证本轮输入不空）
        while total > budget and cut < len(messages) - 1:
            total -= token_list[cut]
            cut += 1
        if cut:
            collector = trace_collector.current()
            if collector is not None:
                collector.record_event(
                    event="truncate",
                    before_tokens=sum(token_list),
                    after_tokens=total,
                    count=cut,
                )
            del messages[:cut]

    @staticmethod
    async def _attach_artifact_refs(db: AsyncSession, messages: list[dict]) -> None:
        """在对应消息后追加产物引用行（引用 ID + 类型 + 摘要关键字段，绝不注入 URL/全文）。

        产物在上下文中的意义是让 LLM 知道该结果存在，可通过工具取详情。
        """
        ids = [int(m["id"]) for m in messages if m.get("id")]
        if not ids:
            return
        try:
            refs = await ai_artifact_service.get_message_artifact_refs(db, ids)
        except Exception as e:  # 产物引用不可用时不影响上下文组装
            logger.warning("加载消息产物引用失败: %s", e)
            return
        if not refs:
            return
        for msg in messages:
            msg_id = msg.get("id")
            lines = ContextManager._build_artifact_ref_lines(
                refs.get(int(msg_id)) if msg_id is not None else None
            )
            if not lines:
                continue
            content = msg.get("content") or ""
            sep = "\n" if content else ""
            msg["content"] = f"{content}{sep}{chr(10).join(lines)}"

    @staticmethod
    def _build_artifact_ref_lines(refs: list[dict] | None) -> list[str]:
        """把单个消息的产物引用列表格式化为引用行（summary 截断 200 字）。"""
        if not refs:
            return []
        lines = []
        for ref in refs:
            art_id = ref.get("id")
            art_type = ref.get("type") or "unknown"
            summary = ref.get("summary") or {}
            summary_text = "" if isinstance(summary, dict) and not summary else str(summary)
            if len(summary_text) > _ARTIFACT_SUMMARY_MAX_LEN:
                summary_text = summary_text[:_ARTIFACT_SUMMARY_MAX_LEN] + "…"
            lines.append(f"[[产物 #{art_id}] {art_type}：{summary_text}]")
        return lines


context_manager = ContextManager()
