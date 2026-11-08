"""消息四种终态 + blocks 落库（completed / failed / canceled / awaiting_input）。

终态判定以 DB 为准：同一条消息只允许一种结论，且内容始终在 blocks 里；
中断态以 awaiting_input + pending_interrupt 表达，绝不以 completed + 空 blocks 冒充完成。
"""

import pytest

from app.infrastructure.sse.sse_event_converter import SseEventConverter
from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_message import SysAiMessage
from tests.stubs.mocks import patch_reasoning_boundaries

pytestmark = pytest.mark.requires_db


class _FailingGraph:
    async def astream(self, *a, **kw):
        raise RuntimeError("模型返回 500 secret=abc123")
        yield  # pragma: no cover

    async def aget_state(self, config):
        return None


async def _seed(db):
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
    asst = SysAiMessage(
        conversation_id=conv.id,
        parent_message_id=user_msg.id,
        role="assistant",
        status="running",
        blocks=[],
    )
    db.add(asst)
    await db.flush()
    conv.current_branch_message_id = asst.id
    await db.flush()
    return conv, user_msg, asst


async def _ends(emitter) -> list[dict]:
    return [data for event, data in emitter.events if event == "message.end"]


async def test_completed_terminal_writes_blocks_and_single_end(db, monkeypatch):
    conv, _user, asst = await _seed(db)
    service, emitter, _ = patch_reasoning_boundaries(monkeypatch)

    await service.run(conv.id, 10, asst.id, "gpt-4o-mini", "s1")

    await db.refresh(asst)
    assert asst.status == "completed"
    assert asst.pending_interrupt is None
    assert [b["kind"] for b in asst.blocks] == ["text"]
    assert asst.text == "ok"
    ends = await _ends(emitter)
    assert len(ends) == 1
    assert ends[0]["status"] == "completed"
    assert ends[0]["messageId"] == asst.id


async def test_failed_terminal_keeps_generated_blocks(db, monkeypatch):
    conv, _user, asst = await _seed(db)
    service, _emitter, _ = patch_reasoning_boundaries(monkeypatch)

    async def _graph(*a, **kw):
        return _FailingGraph()

    monkeypatch.setattr(service, "_build_graph", _graph)

    with pytest.raises(RuntimeError):
        await service.run(conv.id, 10, asst.id, "gpt-4o-mini", "s1")

    await db.refresh(asst)
    assert asst.status == "failed"
    assert "secret=abc123" not in (asst.error or "")
    assert asst.blocks == []


async def test_finalize_canceled_keeps_blocks_and_single_end(db, monkeypatch):
    conv, _user, asst = await _seed(db)
    service, emitter, _ = patch_reasoning_boundaries(monkeypatch)

    async def _settle(*a, **kw):
        return None

    monkeypatch.setattr(
        "app.service.ai.service.reasoning_service.billing_service",
        type("B", (), {"settle": staticmethod(_settle)})(),
    )

    converter = SseEventConverter(
        {"stream_session_id": "s1", "message_id": asst.id, "conversation_id": conv.id}
    )
    from langchain_core.messages import AIMessageChunk

    await converter.handle(
        {"type": "messages", "ns": [], "data": [AIMessageChunk(content="半截回复")]}
    )

    await service._finalize_canceled(
        conv.id,
        10,
        asst.id,
        "gpt-4o-mini",
        {"final_response": "半截回复", "stop_reason": "stop", "usage": {}},
        None,
        "s1",
        converter,
    )

    await db.refresh(asst)
    assert asst.status == "canceled"
    assert asst.text == "半截回复"
    ends = await _ends(emitter)
    assert len(ends) == 1
    assert ends[0]["status"] == "canceled"


async def test_interrupt_terminal_then_resume_clears_pending(db, monkeypatch):
    conv, _user, asst = await _seed(db)
    interrupt = {
        "type": "confirm",
        "data": {"stream_session_id": "s1", "data": {"confirmKind": "tool_permission"}},
    }
    service, _emitter, _ = patch_reasoning_boundaries(monkeypatch, interrupt=interrupt)

    await service.run(conv.id, 10, asst.id, "gpt-4o-mini", "s1")

    await db.refresh(asst)
    assert asst.status == "awaiting_input"
    pending = asst.pending_interrupt
    assert pending["type"] == "confirm"
    assert pending["confirmKind"] == "tool_permission"
    assert pending["actionable"] is True
    assert pending["expiresAt"] > pending["createdAt"]
    # 中断也必须有内容：图 state 的最终答复补成 text 块，不是 completed + 空 blocks
    assert asst.text == "ok"

    # resume 成功：pending 清空、状态回到 completed
    await service._finalize_message(
        asst.id,
        {"final_response": "ok", "stop_reason": "stop", "usage": {}},
        "gpt-4o-mini",
        blocks=asst.blocks,
        pending_interrupt=None,
    )
    await db.refresh(asst)
    assert asst.status == "completed"
    assert asst.pending_interrupt is None
    assert asst.text == "ok"


async def test_expired_interrupt_marked_not_actionable(db):
    from app.service.ai.service.message_view import interrupt_view

    msg = SysAiMessage(
        conversation_id=1,
        role="assistant",
        status="awaiting_input",
        blocks=[],
        pending_interrupt={
            "id": "i1",
            "type": "confirm",
            "createdAt": "2020-01-01T00:00:00",
            "expiresAt": "2020-01-31T00:00:00",
            "actionable": True,
        },
    )
    view = interrupt_view(msg.pending_interrupt)
    assert view is not None
    assert view.actionable is False
    # 无中断时返回 None（GET /messages/{id}/interrupt 出参为 null）
    assert interrupt_view(None) is None


def test_pending_interrupt_carries_ttl_and_identity():
    from app.service.ai.service.contract_payload import build_pending_interrupt

    payload = build_pending_interrupt(
        {"type": "quota", "data": {"upgradeTip": "升级会员", "usedDaily": 3}}
    )
    assert payload["type"] == "quota"
    assert payload["upgradeTip"] == "升级会员"
    assert payload["usedDaily"] == 3
    assert payload["actionable"] is True
    assert payload["id"]
    assert payload["expiresAt"] > payload["createdAt"]


def test_pending_interrupt_time_uses_shanghai_clock():
    """createdAt/expiresAt 与 create_time 同为上海时间，不取宿主本地时间。"""
    from datetime import datetime, timedelta

    from app.models.base import _shanghai_now
    from app.service.ai.service.contract_payload import build_pending_interrupt

    before = _shanghai_now()
    payload = build_pending_interrupt({"type": "confirm", "data": {}})
    after = _shanghai_now()

    created = datetime.fromisoformat(payload["createdAt"])
    expires = datetime.fromisoformat(payload["expiresAt"])
    assert before <= created <= after
    assert expires - created == timedelta(days=30)


def test_plan_revision_time_uses_shanghai_clock():
    from datetime import datetime

    from app.models.base import _shanghai_now
    from app.service.ai.service.contract_payload import to_contract_plan

    before = _shanghai_now()
    plan = to_contract_plan(
        {"tasks": [], "status": "pending", "revisions": [{"reason": "调整依赖"}]}
    )
    after = _shanghai_now()

    at = datetime.fromisoformat(plan["revisions"][0]["at"])
    assert before <= at <= after
