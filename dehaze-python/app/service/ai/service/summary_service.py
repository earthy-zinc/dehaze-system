"""自动摘要压缩服务（SummaryService）

对"被 token 预算窗口裁剪掉、且尚未摘要"的历史消息生成摘要，更新会话 summary。
压缩在推理前同步触发，压缩后的 summary 下轮生效（当前轮上下文已组装完成）。

增量治理：只摘要"上次摘要水位（summary_upto_message_id）之后、当前窗口之前"的
消息，避免每次触发对全部历史全量重摘导致摘要无限膨胀。
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.llm.call.llm_client import llm_client
from app.repository.ai_message_repository import ai_message_repository
from app.repository.ai_model_repository import ai_model_repository
from app.service.ai.service import trace_collector

logger = logging.getLogger(__name__)

# 前序摘要超过该长度时，追加前先对前序摘要自身做一次 LLM 再压缩，防长会话累积膨胀
_PRIOR_SUMMARY_MAX_LEN = 2000
_SUMMARY_PROMPT = (
    "请将以下对话历史压缩为简洁摘要，保留关键信息、决策和任务状态，"
    "保留最近一次处理的算法和参数：\n\n"
)
_RECOMPRESS_PROMPT = (
    "请将以下已经过压缩的对话摘要进一步压缩为更简洁的版本，保留最关键的信息、决策和任务状态：\n\n"
)

# memory-dev 契约（extract_episodic_from_summary）未落地告警去重：仅首次记 error，避免每次摘要刷屏
_EPISODIC_CONTRACT_MISSING_LOGGED = False


def _warn_episodic_contract_missing(conv_id: int) -> None:
    """情景记忆提取契约未落地时显式告警（去重仅报一次，暴露而非静默跳过）"""
    global _EPISODIC_CONTRACT_MISSING_LOGGED
    if _EPISODIC_CONTRACT_MISSING_LOGGED:
        return
    _EPISODIC_CONTRACT_MISSING_LOGGED = True
    logger.error(
        "情景记忆提取契约未落地：memory_extraction.extract_episodic_from_summary 不存在，"
        "摘要后情景记忆提取被跳过(conv=%s)，请对齐 memory-dev 实现",
        conv_id,
    )


class SummaryService:
    """自动摘要压缩服务（单例）"""

    async def maybe_compress(
        self,
        db: AsyncSession,
        conv,
        model_id: str,
        messages: list[dict],
    ) -> None:
        """检查是否需要摘要压缩，需要则执行。

        组装为单点（§3.2）：消息上下文由 reasoning_service.run 调 build_context
        一次组装后传入（窗口裁剪已在此完成），本方法只负责压缩，不再重复
        build_context（避免记忆 touch 副作用翻倍）。压缩后的 summary 下轮生效。
        """
        model = await ai_model_repository.get_by_model_id(db, model_id)
        if not model:
            return
        # 窗口起点：组装结果中最早的消息 id（之前的历史已被 token 预算窗口裁剪）。
        # 存在"水位之后、窗口之外"的消息即触发压缩（摘要保底，§4.1）——旧实现按
        # 组装后 token 占模型窗口 70% 触发，但组装结果已被窗口裁剪、大窗口模型
        # 永远到不了阈值，窗口外的消息被静默丢弃
        window_start_id = next(
            (m["id"] for m in messages if m.get("id") and m.get("role") in ("user", "assistant")),
            None,
        )
        messages_to_summarize = await self._load_messages_to_summarize(db, conv, window_start_id)
        if not messages_to_summarize:
            return
        # 主压缩 + 前序再压缩两次 LLM 调用采集进同一条独立过程链（trace_type=summary）
        async with trace_collector.bypass_span(
            conversation_id=conv.id,
            message_id=None,
            user_id=conv.user_id,
            model_id=model_id,
            trace_type="summary",
        ):
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
        # 摘要完成后提取情景记忆（memory-dev 契约；契约缺失显式 error，运行失败记 warning）
        await self._extract_episodic_memory(db, conv, messages_to_summarize)

    @staticmethod
    async def _extract_episodic_memory(db: AsyncSession, conv, messages: list[dict]) -> None:
        """摘要完成后提取情景记忆（后端实现 §4.2，memory-dev 契约）。

        extract_episodic_from_summary 为 memory-dev 待落地协程（当前仓库未实现），
        故以 getattr 动态取符号；缺失时不得静默跳过，明确记 error 暴露"契约未落地"
        便于跟进对齐（仅首次告警）。
        """
        from app.service.ai.service import memory_extraction

        extract_episodic_from_summary = getattr(
            memory_extraction, "extract_episodic_from_summary", None
        )
        if extract_episodic_from_summary is None:
            _warn_episodic_contract_missing(conv.id)
            return
        try:
            await extract_episodic_from_summary(db, conv.user_id, conv.id, messages)
        except Exception as e:
            logger.warning("情景记忆提取失败(conv=%s): %s", conv.id, e)

    @staticmethod
    async def _load_messages_to_summarize(
        db: AsyncSession, conv, window_start_id: int | None
    ) -> list[dict]:
        """增量加载需要摘要的消息（沿当前激活分支链，避免摘要混入其他分支）。

        只取"摘要水位之后、当前窗口之前"的消息：
        - 上界：summary_upto_message_id（已覆盖范围），未覆盖过则从最早开始
        - 下界：window_start_id（组装窗口最早消息），窗口内保留原文不参与压缩；
          全量历史都在窗口内（无裁剪）时返回空
        """
        watermark = conv.summary_upto_message_id or 0
        chain = await ai_message_repository.get_chain_by_id(
            db, conv.id, getattr(conv, "current_branch_message_id", None), limit=None
        )
        rows = [m for m in chain if m.id > watermark]
        if window_start_id is not None:
            rows = [m for m in rows if m.id < window_start_id]
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
        async for chunk in llm_client.stream_chat(
            db,
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
