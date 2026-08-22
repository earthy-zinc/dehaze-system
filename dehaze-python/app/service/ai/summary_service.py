"""自动摘要压缩服务（SummaryService）

当会话上下文 token 超过模型上下文阈值 70% 时，对老消息生成摘要并更新会话 summary。
摘要压缩在推理之前执行，确保当前推理使用的上下文是已压缩的。

增量治理：只摘要"上次摘要水位（summary_upto_message_id）之后、最近 N 轮之前"的
消息，避免每次触发对全部历史全量重摘导致摘要无限膨胀。
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.redis import get_redis_client
from app.repository.ai_message_repository import ai_message_repository
from app.repository.ai_model_repository import ai_model_repository
from app.service.ai.context_manager import (
    _RECENT_MESSAGE_LIMIT,
    context_manager,
    estimate_context_tokens,
)
from app.infrastructure.llm.llm_client import llm_client

logger = logging.getLogger(__name__)

# 触发摘要压缩的上下文占用阈值（模型 max_context_tokens 的比例）
_SUMMARY_THRESHOLD_RATIO = 0.7
# 前序摘要超过该长度时，追加前先对前序摘要自身做一次 LLM 再压缩，防长会话累积膨胀
_PRIOR_SUMMARY_MAX_LEN = 2000
_SUMMARY_PROMPT = (
    "请将以下对话历史压缩为简洁摘要，保留关键信息、决策和任务状态，"
    "保留最近一次处理的算法和参数：\n\n"
)
_RECOMPRESS_PROMPT = (
    "请将以下已经过压缩的对话摘要进一步压缩为更简洁的版本，保留最关键的信息、决策和任务状态：\n\n"
)


class SummaryService:
    """自动摘要压缩服务（单例）"""

    async def maybe_compress(self, db: AsyncSession, conv, model_id: str) -> None:
        """检查是否需要摘要压缩，需要则执行"""
        model = await ai_model_repository.get_by_model_id(db, model_id)
        if not model:
            return
        messages, system_prompt, _injected = await context_manager.build_context(db, conv, model_id)
        total_tokens = await estimate_context_tokens(messages, system_prompt)
        threshold = int(model.max_context_tokens * _SUMMARY_THRESHOLD_RATIO)
        if total_tokens <= threshold:
            return
        messages_to_summarize = await self._load_messages_to_summarize(db, conv)
        if not messages_to_summarize:
            return
        new_content = await self._generate_summary(db, model_id, messages_to_summarize)
        if not new_content:
            return
        old_summary = conv.summary
        if old_summary:
            if len(old_summary) > _PRIOR_SUMMARY_MAX_LEN:
                old_summary = await self._recompress_prior_summary(db, model_id, old_summary)
            conv.summary = f"前序摘要：{old_summary}\n近期摘要：{new_content}"
        else:
            conv.summary = new_content
        # 推进摘要水位到本次覆盖的最后一条消息
        conv.summary_upto_message_id = messages_to_summarize[-1]["id"]
        await db.flush()
        # 摘要完成后提取情景记忆（memory-dev 契约；失败静默记日志）
        await self._extract_episodic_memory(db, conv, messages_to_summarize)

    @staticmethod
    async def _extract_episodic_memory(db: AsyncSession, conv, messages: list[dict]) -> None:
        try:
            from app.service.ai.memory_extraction import extract_episodic_from_summary
        except ImportError:
            return
        try:
            await extract_episodic_from_summary(db, conv.user_id, conv.id, messages)
        except Exception as e:
            logger.warning("情景记忆提取失败(conv=%s): %s", conv.id, e)

    @staticmethod
    async def _load_messages_to_summarize(db: AsyncSession, conv) -> list[dict]:
        """增量加载需要摘要的消息。

        只取"摘要水位之后、最近 N 轮之前"的消息：
        - 上界：summary_upto_message_id（已覆盖范围），未覆盖过则从最早开始
        - 下界：最近 _RECENT_MESSAGE_LIMIT 条保留原文，不参与压缩
        """
        watermark = conv.summary_upto_message_id or 0
        rows = await ai_message_repository.list_for_summary(db, conv.id, watermark)
        # 去掉最近 N 轮（按时间倒序的最前面），其余反转成正序
        rows = list(reversed(rows[_RECENT_MESSAGE_LIMIT:]))
        return [
            {"id": m.id, "role": m.role, "content": m.content}
            for m in rows
            if m.role in ("user", "assistant") and m.content
        ]

    @staticmethod
    async def _generate_summary(
        db: AsyncSession,
        model_id: str,
        messages_to_summarize: list[dict],
    ) -> str:
        """调用 LLM 生成摘要（非流式，只收集完整内容）"""
        history = "\n".join(f"{m['role']}: {m['content']}" for m in messages_to_summarize)
        return await SummaryService._run_llm(db, model_id, _SUMMARY_PROMPT + history)

    @staticmethod
    async def _recompress_prior_summary(
        db: AsyncSession,
        model_id: str,
        old_summary: str,
    ) -> str:
        """对过长前序摘要自身做一次 LLM 再压缩，防长会话累积膨胀。"""
        compressed = await SummaryService._run_llm(db, model_id, _RECOMPRESS_PROMPT + old_summary)
        return compressed or old_summary

    @staticmethod
    async def _run_llm(db: AsyncSession, model_id: str, content: str) -> str:
        """调用 LLM 生成/压缩文本（非流式，只收集完整内容，temperature=0）"""
        out = ""
        redis = await get_redis_client()
        async for chunk in llm_client.stream_chat(
            db,
            redis,
            model_id,
            [{"role": "user", "content": content}],
            system_prompt="你是对话摘要助手",
            temperature=0,
            max_tokens=500,
        ):
            if chunk.type == "text_delta":
                out += chunk.content
        return out


summary_service = SummaryService()
