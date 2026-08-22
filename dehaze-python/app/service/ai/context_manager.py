"""上下文组装器（ContextManager）

每次推理前组装发送给 LLM 的消息列表：
system_prompt → summary（system 消息）→ 长期记忆 → 最近 N 轮原文（含产物引用行）。
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.ai_message_repository import ai_message_repository
from app.service.ai.llm_client import llm_client
from app.service.ai.memory_injection import inject_memories
from app.service.ai.prompt_composer import compose_system_prompt
from app.service.ai_artifact_service import ai_artifact_service

logger = logging.getLogger(__name__)

# 保留最近 N 轮原文（不参与摘要压缩）
_RECENT_MESSAGE_LIMIT = 20
# 产物引用行中 summary 字段的最大展示长度
_ARTIFACT_SUMMARY_MAX_LEN = 200


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
        messages = await self._load_recent_messages(db, conv)
        # 关联产物引用行（按 message_id 查询，仅在对应消息后追加引用，绝不注入全文/URL）
        await self._attach_artifact_refs(db, messages)
        if conv.summary:
            messages.insert(0, {"role": "system", "content": f"之前的对话摘要：{conv.summary}"})
        # 长期记忆注入（在 summary 之后、对话消息之前）：
        # inject_memories 返回 (system_block_text, injected_list)，
        # system 补充块作为 system 消息注入。
        last_user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user" and msg["content"]:
                last_user_msg = msg["content"]
                break
        system_block, injected_list = await inject_memories(db, conv.user_id, last_user_msg)
        if system_block:
            memory_msg = {"role": "system", "content": system_block}
            insert_at = 1 if conv.summary else 0
            messages[insert_at:insert_at] = [memory_msg]
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
    async def _load_recent_messages(db: AsyncSession, conv) -> list[dict]:
        """查询最近 N 轮消息（沿 current_branch_message_id 的分支链回溯）。

        过滤 deleted=1，只取 user/assistant 且 content 非空，按时间正序。
        保留 message_id 以支撑产物引用关联。
        """
        msgs = await ai_message_repository.get_chain_by_id(
            db,
            conv.id,
            conv.current_branch_message_id,
            limit=_RECENT_MESSAGE_LIMIT,
        )
        return [
            {"id": m.id, "role": m.role, "content": m.content}
            for m in msgs
            if m.role in ("user", "assistant") and m.content
        ]

    @staticmethod
    async def _attach_artifact_refs(db: AsyncSession, messages: list[dict]) -> None:
        """在对应消息后追加产物引用行（引用 ID + 类型 + 摘要关键字段，绝不注入 URL/全文）。

        产物在上下文中的意义是让 LLM 知道该结果存在，可通过工具取详情。
        """
        ids = [m.get("id") for m in messages if m.get("id")]
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
            lines = ContextManager._build_artifact_ref_lines(refs.get(msg.get("id")))
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


async def estimate_context_tokens(messages: list[dict], system_prompt: str | None) -> int:
    """估算上下文 token 总量"""
    total = 0
    for msg in messages:
        total += await llm_client.count_tokens(msg.get("content", ""))
    if system_prompt:
        total += await llm_client.count_tokens(system_prompt)
    return total


context_manager = ContextManager()
