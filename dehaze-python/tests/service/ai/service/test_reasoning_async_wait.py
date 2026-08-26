import pytest

from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_message import SysAiMessage
from tests.stubs.mocks import patch_reasoning_boundaries

pytestmark = pytest.mark.requires_db


async def _seed_run_ctx(db):
    """落库真实会话 + 用户消息 + 待生成的 assistant 消息（status=1 生成中）"""
    conv = SysAiConversation(user_id=10, model="gpt-4o-mini")
    db.add(conv)
    await db.flush()
    user_msg = SysAiMessage(
        conversation_id=conv.id, parent_message_id=None, role="user", content="帮我处理雾图", status=2
    )
    db.add(user_msg)
    await db.flush()
    asst_msg = SysAiMessage(
        conversation_id=conv.id,
        parent_message_id=user_msg.id,
        role="assistant",
        content="",
        status=1,
    )
    db.add(asst_msg)
    await db.flush()
    conv.current_branch_message_id = asst_msg.id
    await db.flush()
    return conv, asst_msg


async def test_async_wait_suspend_skips_finalize_but_pushes_end(db, monkeypatch):
    interrupt_data = {
        "type": "async_wait",
        "data": {"task_id": "batch:1:2:1", "stream_session_id": "s1"},
    }
    conv, asst_msg = await _seed_run_ctx(db)
    service, emitter, _ = patch_reasoning_boundaries(monkeypatch, interrupt=interrupt_data)

    result = await service.run(conv.id, 10, asst_msg.id, "gpt-4o-mini", "s1")

    # 业务结果：async_wait 挂起不落库最终态（消息保持"生成中"，待 resume 续写）
    await db.refresh(asst_msg)
    assert asst_msg.status == 1
    assert asst_msg.content == ""
    # 释放会话并发锁让渡给 resume；message.end 以 0 积分收尾
    assert emitter.released == [conv.id]
    assert emitter.events[-1][0] == "message.end"
    assert emitter.events[-1][1]["usage"]["credits"] == 0
    assert result["stop_reason"] == "stop"


async def test_confirm_suspend_still_finalizes(db, monkeypatch):
    interrupt_data = {"type": "confirm", "data": {"stream_session_id": "s1"}}
    conv, asst_msg = await _seed_run_ctx(db)
    service, emitter, _ = patch_reasoning_boundaries(monkeypatch, interrupt=interrupt_data)

    await service.run(conv.id, 10, asst_msg.id, "gpt-4o-mini", "s1")

    # 业务结果：confirm 中断仅让渡锁，仍完成落库（内容/token/状态）
    await db.refresh(asst_msg)
    assert asst_msg.status == 2
    assert asst_msg.content == "ok"
    assert asst_msg.input_tokens == 5
    assert asst_msg.output_tokens == 3
    assert emitter.released == [conv.id]
    assert emitter.events[-1][0] == "message.end"
