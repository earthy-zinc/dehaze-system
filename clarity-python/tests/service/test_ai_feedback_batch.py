"""会话级批量反馈：一次查询返回当前用户在该会话下的全部未删反馈。

对应 G6：进入会话时前端不再逐条消息拉反馈，而是一次请求回填；权限与单条反馈接口
一致（会话必须属于当前用户），且只返回本人数据。
"""

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_message import SysAiMessage
from app.models.entity.sys_ai_message_feedback import SysAiMessageFeedback
from app.repository.ai_message_feedback_repository import ai_message_feedback_repository
from app.service.ai_feedback_service import ai_feedback_service

pytestmark = pytest.mark.requires_db


async def _seed_conv(db, user_id=1) -> SysAiConversation:
    conv = SysAiConversation(user_id=user_id, model="gpt", agent_code=None)
    db.add(conv)
    await db.flush()
    return conv


async def _seed_msg(db, conv_id) -> SysAiMessage:
    msg = SysAiMessage(
        conversation_id=conv_id,
        parent_message_id=None,
        role="assistant",
        status="completed",
        blocks=[],
    )
    db.add(msg)
    await db.flush()
    return msg


async def _seed_feedback(db, msg_id, conv_id, user_id, rating=1, **overrides):
    fields = {
        "message_id": msg_id,
        "conversation_id": conv_id,
        "user_id": user_id,
        "rating": rating,
        "tags": None,
        "comment": None,
        "source": "internal",
        "processed": 0,
    }
    fields.update(overrides)
    feedback = SysAiMessageFeedback(**fields)
    db.add(feedback)
    await db.flush()
    return feedback


async def test_returns_current_user_feedback_in_conversation(db):
    conv = await _seed_conv(db)
    m1 = await _seed_msg(db, conv.id)
    m2 = await _seed_msg(db, conv.id)
    like = await _seed_feedback(db, m1.id, conv.id, 1, rating=1, tags=["accurate"])
    dislike = await _seed_feedback(
        db, m2.id, conv.id, 1, rating=-1, tags=["too_long"], comment="太啰嗦"
    )
    # 他人反馈、其他会话反馈、已撤销反馈都不应出现在结果里
    await _seed_feedback(db, m2.id, conv.id, 2, rating=1)
    other_conv = await _seed_conv(db)
    other_msg = await _seed_msg(db, other_conv.id)
    await _seed_feedback(db, other_msg.id, other_conv.id, 1, rating=1)
    revoked_msg = await _seed_msg(db, conv.id)
    await _seed_feedback(db, revoked_msg.id, conv.id, 1, rating=1)
    await ai_message_feedback_repository.soft_delete(db, revoked_msg.id, 1)

    rows = await ai_feedback_service.list_conversation_feedback(db, conv.id, 1)

    assert [row.messageId for row in rows] == [like.message_id, dislike.message_id]
    assert [row.id for row in rows] == [like.id, dislike.id]
    assert all(row.userId == 1 and row.conversationId == conv.id for row in rows)
    assert rows[0].rating == 1
    assert rows[0].tags == ["accurate"]
    assert rows[1].rating == -1
    assert rows[1].tags == ["too_long"]
    assert rows[1].comment == "太啰嗦"
    assert rows[0].createTime is not None


async def test_empty_conversation_returns_empty_list(db):
    conv = await _seed_conv(db)

    assert await ai_feedback_service.list_conversation_feedback(db, conv.id, 1) == []


async def test_missing_conversation_raises_a0401(db):
    with pytest.raises(BusinessException) as ei:
        await ai_feedback_service.list_conversation_feedback(db, 999999, 1)
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


async def test_other_users_conversation_raises_a0401(db):
    conv = await _seed_conv(db, user_id=99)
    msg = await _seed_msg(db, conv.id)
    await _seed_feedback(db, msg.id, conv.id, 99, rating=1)

    with pytest.raises(BusinessException) as ei:
        await ai_feedback_service.list_conversation_feedback(db, conv.id, 1)
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND
