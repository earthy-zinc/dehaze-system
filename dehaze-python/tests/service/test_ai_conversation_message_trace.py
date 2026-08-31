"""消息详情可观测性扩展测试：trace_id / context_snapshot / llm_calls 附带"""

import pytest

from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_llm_call import SysAiLlmCall
from app.models.entity.sys_ai_message import SysAiMessage
from app.repository.ai_llm_call_repository import ai_llm_call_repository
from app.repository.ai_trace_repository import ai_trace_repository
from app.service.ai_conversation_service import ai_conversation_service

pytestmark = pytest.mark.requires_db


async def _make_fixture(db, with_trace: bool):
    conv = SysAiConversation(user_id=77, title="t")
    db.add(conv)
    await db.flush()
    msg = SysAiMessage(
        conversation_id=conv.id, role="assistant", content="answer", status=2
    )
    db.add(msg)
    await db.flush()
    if with_trace:
        await ai_trace_repository.insert_idempotent(
            db,
            {
                "trace_id": "tr-msg",
                "conversation_id": conv.id,
                "message_id": msg.id,
                "status": 1,
                "duration_ms": 80,
                "llm_call_count": 2,
                "total_tokens": 9,
                "prompt_tokens": 6,
                "completion_tokens": 3,
                "cached_tokens": 1,
                "step_count": 2,
                "context_snapshot": {"items": [{"type": "system", "tokens": 3}], "events": []},
            },
        )
        for seq in (1, 2):
            await ai_llm_call_repository.insert_idempotent(
                db,
                {
                    "trace_id": "tr-msg",
                    "seq": seq,
                    "step_position": seq,
                    "model": "gpt-x",
                    "status": 1,
                    "duration_ms": 10,
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "cached_tokens": 0,
                },
            )
    return conv, msg


async def test_get_message_attaches_trace_and_llm_calls(db):
    conv, msg = await _make_fixture(db, with_trace=True)
    result = await ai_conversation_service.get_message(db, msg.id, conv.user_id)
    assert result["traceId"] == "tr-msg"
    assert result["contextSnapshot"]["items"] == [{"type": "system", "tokens": 3}]
    assert [c["seq"] for c in result["llmCalls"]] == [1, 2]  # 按 seq 正序回放
    assert result["llmCalls"][0]["model"] == "gpt-x"


async def test_get_message_without_trace_returns_empty_observability(db):
    conv, msg = await _make_fixture(db, with_trace=False)
    result = await ai_conversation_service.get_message(db, msg.id, conv.user_id)
    assert result["traceId"] is None
    assert result["contextSnapshot"] is None
    assert result["llmCalls"] == []
