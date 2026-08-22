"""类似问题推荐服务（SuggestionService）

回复完成后为引导追问，由 LLM 生成 2-3 条推荐问题，经 SSE suggestions 事件推送
（设计文档 §4.7 对齐 ChatGPT 回复末尾的"相关问题"交互）。

行为约定：
- 会话设置 model_config.suggest_questions 关闭时跳过（默认开启）
- 异步触发，失败/超时记 warning 日志后跳过，不阻塞主回复的 message.end
- 生成成功：推荐问题 Token 计入该条回复消耗（与回复同一条计费记录，不单独计费）
"""

import asyncio
import json
import logging
import re

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db_session
from app.dependencies.redis import get_redis_client
from app.infrastructure.sse.sse_emitter_manager import sse_emitter_manager
from app.repository.ai_conversation_repository import ai_conversation_repository
from app.repository.ai_message_repository import ai_message_repository
from app.service.ai.credits_service import calculate_credits
from app.service.ai.llm_client import llm_client
from app.service.billing.billing_service import billing_service

logger = logging.getLogger(__name__)

# 生成推荐问题的短超时（秒）：后台任务，超时记日志跳过
_SUGGESTION_TIMEOUT = 10
# 推荐问题数量范围
_SUGGESTION_MIN = 2
_SUGGESTION_MAX = 3
# 推荐问题生成提示词（要求返回 JSON 数组）
_SUGGESTION_PROMPT = (
    "请基于 AI 的回答生成 {min}-{max} 条与回答内容相关、口语化的中文追问建议，"
    '直接返回 JSON 数组，例如 ["追问一", "追问二"]，不要输出任何其他文字。\n\n'
    "AI 回答：\n{reply}"
)

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class SuggestionService:
    """类似问题推荐服务（单例）"""

    @staticmethod
    async def _is_enabled(conv) -> bool:
        """会话设置开关：model_config.suggest_questions，默认开启。"""
        return bool((conv.model_config or {}).get("suggest_questions", True))

    @staticmethod
    async def _parse_questions(content: str) -> list[str] | None:
        """从 LLM 输出解析推荐问题 JSON 数组，解析失败返回 None。"""
        if not content:
            return None
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            match = _JSON_ARRAY_RE.search(content)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
            except (json.JSONDecodeError, TypeError):
                return None
        if not isinstance(data, list):
            return None
        questions = [str(q).strip() for q in data if str(q).strip()]
        return questions if questions else None

    async def _generate_questions(
        self, db: AsyncSession, model_id: str, reply_content: str
    ) -> tuple[list[str], dict] | None:
        """调用 LLM 生成推荐问题，返回 (questions, usage)；失败/超时返回 None。"""
        redis = await get_redis_client()
        content = ""
        usage: dict = {}
        try:

            async def _collect() -> None:
                nonlocal content, usage
                async for chunk in llm_client.stream_chat(
                    db,
                    redis,
                    model_id,
                    [
                        {
                            "role": "user",
                            "content": _SUGGESTION_PROMPT.format(
                                min=_SUGGESTION_MIN, max=_SUGGESTION_MAX, reply=reply_content
                            ),
                        }
                    ],
                    system_prompt="你是对话助手，负责生成推荐追问",
                    temperature=0.7,
                    max_tokens=200,
                ):
                    if chunk.type == "text_delta":
                        content += chunk.content
                    elif chunk.type == "done" and chunk.usage:
                        usage = chunk.usage

            await asyncio.wait_for(_collect(), timeout=_SUGGESTION_TIMEOUT)
        except TimeoutError:
            logger.warning("类似问题推荐生成超时（%ss），跳过", _SUGGESTION_TIMEOUT)
            return None
        except Exception:
            logger.warning("类似问题推荐生成失败", exc_info=True)
            return None

        questions = await self._parse_questions(content)
        if not questions:
            return None
        return questions, usage

    async def generate(
        self,
        conversation_id: int,
        message_id: int,
        reply_content: str,
        user_id: int,
        stream_session_id: str,
    ) -> list[str] | None:
        """为指定回复生成推荐问题并计入该条回复消耗，成功返回问题列表。

        会话设置关闭、生成失败/超时、计费补充失败均记日志跳过（不阻塞 message.end）。
        """
        if not reply_content or not stream_session_id:
            return None

        async with get_db_session() as db:
            conv = await ai_conversation_repository.get_by_id(db, conversation_id)
            if not conv or not await self._is_enabled(conv):
                return None
            msg = await ai_message_repository.get_by_id(db, message_id)
            if not msg:
                return None
            model_id = msg.model or conv.model or settings.AI_DEFAULT_MODEL

            generated = await self._generate_questions(db, model_id, reply_content)
            if not generated:
                return None
            questions, sugg_usage = generated

            # 推荐问题 Token 计入该条回复：累加 token 字段与积分
            new_input = msg.input_tokens + int(sugg_usage.get("input_tokens", 0))
            new_output = msg.output_tokens + int(sugg_usage.get("output_tokens", 0))
            new_cached = msg.cached_input_tokens + int(sugg_usage.get("cached_input_tokens", 0))
            msg.input_tokens = new_input
            msg.output_tokens = new_output
            msg.cached_input_tokens = new_cached
            msg.credits = await calculate_credits(db, model_id, new_input, new_output, new_cached)
            await db.flush()

            # 补充结算：与回复同一条计费记录（settle 按 message_id 定位并差额退补）。
            # adjustment=True：仅退补差额、更新记录，不新增 consume 流水、不做异常检测
            try:
                await billing_service.settle(
                    db,
                    user_id,
                    conversation_id,
                    message_id,
                    model_id,
                    None,
                    {
                        "input_tokens": new_input,
                        "output_tokens": new_output,
                        "cached_input_tokens": new_cached,
                    },
                    adjustment=True,
                )
            except Exception:
                logger.warning("类似问题推荐计费补充失败", exc_info=True)

        # 推送 suggestions 事件（异步触发方保证 message.end 已推送）
        await sse_emitter_manager.send_event(
            stream_session_id, "suggestions", {"questions": [{"question": q} for q in questions]}
        )
        return questions


suggestion_service = SuggestionService()
