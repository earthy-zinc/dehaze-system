import pytest

from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_message import SysAiMessage
from tests.stubs.mocks import patch_reasoning_boundaries

pytestmark = pytest.mark.requires_db


async def _seed_run_ctx(db):
    """落库真实会话 + 用户消息 + 待生成的 assistant 消息（status=running）"""
    conv = SysAiConversation(user_id=10, model="gpt-4o-mini")
    db.add(conv)
    await db.flush()
    user_msg = SysAiMessage(
        conversation_id=conv.id,
        parent_message_id=None,
        role="user",
        status="completed",
        blocks=[{"id": "u1", "kind": "text", "seq": 1, "text": "帮我处理雾图"}],
    )
    db.add(user_msg)
    await db.flush()
    asst_msg = SysAiMessage(
        conversation_id=conv.id,
        parent_message_id=user_msg.id,
        role="assistant",
        status="running",
        blocks=[],
    )
    db.add(asst_msg)
    await db.flush()
    conv.current_branch_message_id = asst_msg.id
    await db.flush()
    return conv, asst_msg


async def test_async_wait_suspend_skips_finalize_but_pushes_end(db, monkeypatch):
    interrupt_data = {
        "type": "async_wait",
        "data": {"taskId": "batch:1:2:1", "stream_session_id": "s1"},
    }
    conv, asst_msg = await _seed_run_ctx(db)
    service, emitter, _ = patch_reasoning_boundaries(monkeypatch, interrupt=interrupt_data)

    result = await service.run(conv.id, 10, asst_msg.id, "gpt-4o-mini", "s1")

    # 业务结果：async_wait 挂起，消息停在 awaiting_input 并持久化中断（可刷新后恢复）
    await db.refresh(asst_msg)
    assert asst_msg.status == "awaiting_input"
    assert asst_msg.pending_interrupt["type"] == "async_wait"
    assert asst_msg.pending_interrupt["actionable"] is True
    assert asst_msg.pending_interrupt["id"]
    assert asst_msg.pending_interrupt["expiresAt"]
    # 释放会话并发锁让渡给 resume；终帧以 awaiting_input 收尾
    assert emitter.released == [conv.id]
    assert emitter.events[-1][0] == "message.end"
    assert emitter.events[-1][1]["status"] == "awaiting_input"
    assert result["stop_reason"] == "stop"


async def test_confirm_suspend_still_finalizes(db, monkeypatch):
    interrupt_data = {"type": "confirm", "data": {"stream_session_id": "s1"}}
    conv, asst_msg = await _seed_run_ctx(db)
    service, emitter, _ = patch_reasoning_boundaries(monkeypatch, interrupt=interrupt_data)

    await service.run(conv.id, 10, asst_msg.id, "gpt-4o-mini", "s1")

    # 业务结果：confirm 中断让渡锁，内容/token 照常落库，状态为等待确认
    await db.refresh(asst_msg)
    assert asst_msg.status == "awaiting_input"
    assert asst_msg.text == "ok"
    assert asst_msg.pending_interrupt["type"] == "confirm"
    assert asst_msg.input_tokens == 5
    assert asst_msg.output_tokens == 3
    assert emitter.released == [conv.id]
    assert emitter.events[-1][0] == "message.end"
    assert emitter.events[-1][1]["status"] == "awaiting_input"
