"""AI 对话模块 - 消息反馈服务"""

import asyncio
import logging
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.schema.ai_feedback import FeedbackCreateRequest, FeedbackResult
from app.repository.ai_message_feedback_repository import ai_message_feedback_repository
from app.repository.ai_message_repository import ai_message_repository

logger = logging.getLogger(__name__)

FEEDBACK_VALID_DAYS = 30

# 标签白名单（对齐需求规格 §2.9.3）
LIKE_TAGS = {"accurate", "detailed", "concise", "creative"}
DISLIKE_TAGS = {"incorrect", "irrelevant", "incomplete", "too_long", "bad_citation", "harmful"}

# 点踩标签 → 偏好语义记忆（source=feedback，is_preference=1），供记忆注入常驻生效
_DISLIKE_PREFERENCE_MEMORIES = {
    "too_long": "用户偏好简洁回复",
    "incomplete": "用户期望回复完整、覆盖全部要点",
    "irrelevant": "用户期望回复紧扣主题、避免无关内容",
}

# 异步后台任务引用，防止被垃圾回收
_pending_tasks: set[asyncio.Task] = set()


def _spawn_feedback_memory_extraction(
    user_id: int,
    tags: list[str] | None,
    comment: str | None,
) -> None:
    """异步将点踩标签沉淀为用户偏好语义记忆（不阻塞反馈提交）。"""
    preference = _build_preference_content(tags, comment)
    if not preference:
        return

    async def _run() -> None:
        from app.database import get_db_session
        from app.models.entity.sys_ai_memory import SysAiMemory
        from app.repository.ai_memory_repository import ai_memory_repository

        try:
            async with get_db_session() as db:
                memory = SysAiMemory(
                    user_id=user_id,
                    memory_type="semantic",
                    content=preference,
                    metadata_={"category": "preference", "is_preference": 1},
                    importance=100,
                    source="feedback",
                    status=1,
                    archived=0,
                )
                await ai_memory_repository.create(db, memory)
        except Exception:  # noqa: BLE001 - 反馈记忆沉淀失败不影响主流程
            logger.warning("反馈记忆沉淀失败 user_id=%s", user_id, exc_info=True)

    task = asyncio.create_task(_run())
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)


def _build_preference_content(tags: list[str] | None, comment: str | None) -> str | None:
    """由点踩标签生成用户偏好内容；优先取已映射标签，再结合用户补充意见。"""
    mapped = None
    for tag in tags or []:
        if tag in _DISLIKE_PREFERENCE_MEMORIES:
            mapped = _DISLIKE_PREFERENCE_MEMORIES[tag]
            break
    if not mapped:
        return None
    if comment and comment.strip():
        return f"{mapped}（用户补充：{comment.strip()}）"
    return mapped


class AiFeedbackService:
    async def submit_feedback(
        self,
        db: AsyncSession,
        message_id: int,
        user_id: int,
        form: FeedbackCreateRequest,
    ) -> FeedbackResult:
        message = await ai_message_repository.get_by_id_and_user(db, message_id, user_id)
        if not message:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在")
        if message.role != "assistant":
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "仅助手消息可反馈")
        if message.create_time and datetime.now() - message.create_time > timedelta(
            days=FEEDBACK_VALID_DAYS
        ):
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "已超过反馈时效(30天)")
        if form.rating == 1:
            if form.tags and not set(form.tags).issubset(LIKE_TAGS):
                raise BusinessException(ResultCode.PARAM_ERROR, "不支持的标签类型")
        else:
            if not form.tags:
                raise BusinessException(ResultCode.PARAM_ERROR, "点踩必须选择问题标签")
            if not set(form.tags).issubset(DISLIKE_TAGS):
                raise BusinessException(ResultCode.PARAM_ERROR, "不支持的标签类型")
        feedback = await ai_message_feedback_repository.upsert_feedback(
            db,
            message_id,
            user_id,
            form.rating,
            form.tags,
            form.comment,
            conversation_id=message.conversation_id,
            model=message.model,
            source="internal",
        )
        # 点踩反馈异步沉淀为用户偏好记忆（source=feedback，is_preference=1），不阻塞反馈提交
        if form.rating != 1:
            _spawn_feedback_memory_extraction(user_id, form.tags, form.comment)
        return FeedbackResult.model_validate(feedback)

    async def get_feedback(
        self,
        db: AsyncSession,
        message_id: int,
        user_id: int,
    ) -> FeedbackResult | None:
        feedback = await ai_message_feedback_repository.get_by_user_and_message(
            db, message_id, user_id
        )
        if not feedback:
            return None
        return FeedbackResult.model_validate(feedback)

    async def revoke_feedback(
        self,
        db: AsyncSession,
        message_id: int,
        user_id: int,
    ) -> None:
        feedback = await ai_message_feedback_repository.get_by_user_and_message(
            db, message_id, user_id
        )
        if not feedback:
            raise BusinessException(ResultCode.FEEDBACK_NOT_FOUND, "反馈不存在")
        await ai_message_feedback_repository.soft_delete(db, message_id, user_id)


ai_feedback_service = AiFeedbackService()
