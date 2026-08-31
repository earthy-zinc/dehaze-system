"""ai_trace / ai_llm_call repository 真实 SQL 语义测试（MySQL dehaze_test）"""

from datetime import datetime

import pytest

from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_llm_call import SysAiLlmCall
from app.models.entity.sys_ai_trace import SysAiTrace
from app.repository.ai_llm_call_repository import ai_llm_call_repository
from app.repository.ai_trace_repository import ai_trace_repository

pytestmark = pytest.mark.requires_db


def _trace_values(trace_id: str, conv_id: int, **overrides) -> dict:
    values = {
        "trace_id": trace_id,
        "conversation_id": conv_id,
        "message_id": None,
        "status": 1,
        "duration_ms": 100,
        "llm_call_count": 1,
        "total_tokens": 30,
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "cached_tokens": 5,
        "step_count": 2,
    }
    values.update(overrides)
    return values


async def _make_conversation(db, user_id: int = 1001) -> SysAiConversation:
    conv = SysAiConversation(user_id=user_id, title="t")
    db.add(conv)
    await db.flush()
    return conv


async def test_insert_idempotent_ignores_duplicate_trace_id(db):
    await ai_trace_repository.insert_idempotent(db, _trace_values("tr-1", 1, status=1))
    await ai_trace_repository.insert_idempotent(db, _trace_values("tr-1", 1, status=2))
    trace = await ai_trace_repository.get_by_trace_id(db, "tr-1")
    assert trace is not None
    assert trace.status == 1  # 重复写入被忽略，保留首条


async def test_list_traces_filters(db):
    conv = await _make_conversation(db, user_id=2002)
    await ai_trace_repository.insert_idempotent(
        db, _trace_values("tr-a", conv.id, status=1)
    )
    await ai_trace_repository.insert_idempotent(
        db, _trace_values("tr-b", conv.id, status=2, agent_code="writer")
    )
    await ai_trace_repository.insert_idempotent(
        db, _trace_values("tr-c", conv.id, status=3, create_time=datetime(2020, 1, 1))
    )

    items, total = await ai_trace_repository.list_traces(db, conversation_id=conv.id)
    assert total == 3

    items, total = await ai_trace_repository.list_traces(db, status=2)
    assert {t.trace_id for t in items} == {"tr-b"}

    items, total = await ai_trace_repository.list_traces(db, user_id=2002)
    assert total == 3  # 用户维度经会话表关联
    _, total = await ai_trace_repository.list_traces(db, user_id=999999)
    assert total == 0

    items, total = await ai_trace_repository.list_traces(db, agent_code="writer")
    assert {t.trace_id for t in items} == {"tr-b"}

    # 时间范围过滤：只命中 2020 年前的 tr-c
    items, total = await ai_trace_repository.list_traces(
        db, end_time=datetime(2021, 1, 1)
    )
    assert {t.trace_id for t in items} == {"tr-c"}

    # 分页
    items, total = await ai_trace_repository.list_traces(
        db, conversation_id=conv.id, page=1, size=2
    )
    assert len(items) == 2 and total == 3


async def test_get_latest_by_message_id(db):
    await ai_trace_repository.insert_idempotent(
        db, _trace_values("tr-old", 1, message_id=55, create_time=datetime(2020, 1, 1))
    )
    await ai_trace_repository.insert_idempotent(
        db, _trace_values("tr-new", 1, message_id=55, status=3)
    )
    trace = await ai_trace_repository.get_latest_by_message_id(db, 55)
    assert trace.trace_id == "tr-new"  # resume 场景中断+成功两条，详情取最新
    assert await ai_trace_repository.get_latest_by_message_id(db, 666) is None


async def test_list_abnormal_conversation_ids(db):
    conv_ok = await _make_conversation(db)
    conv_bad = await _make_conversation(db)
    await ai_trace_repository.insert_idempotent(
        db, _trace_values("tr-ok", conv_ok.id, status=1)
    )
    await ai_trace_repository.insert_idempotent(
        db, _trace_values("tr-bad1", conv_bad.id, status=2)
    )
    await ai_trace_repository.insert_idempotent(
        db, _trace_values("tr-bad2", conv_bad.id, status=4)
    )
    abnormal = await ai_trace_repository.list_abnormal_conversation_ids(
        db, [conv_ok.id, conv_bad.id]
    )
    assert abnormal == {conv_bad.id}
    assert await ai_trace_repository.list_abnormal_conversation_ids(db, []) == set()


async def test_count_by_status(db):
    await ai_trace_repository.insert_idempotent(db, _trace_values("tr-s1", 1, status=1))
    await ai_trace_repository.insert_idempotent(db, _trace_values("tr-s2", 2, status=2))
    await ai_trace_repository.insert_idempotent(db, _trace_values("tr-s3", 3, status=2))
    counts = await ai_trace_repository.count_by_status(db)
    assert counts.get(2) == 2


async def test_llm_call_insert_and_list_by_trace(db):
    for seq in (1, 2):
        await ai_llm_call_repository.insert_idempotent(
            db,
            {
                "trace_id": "tr-llm",
                "seq": seq,
                "step_position": seq,
                "model": "gpt-x",
                "status": 1,
                "duration_ms": 10,
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "cached_tokens": 1,
            },
        )
    # 重复 (trace_id, seq) 幂等忽略
    await ai_llm_call_repository.insert_idempotent(
        db,
        {
            "trace_id": "tr-llm",
            "seq": 1,
            "step_position": None,
            "model": "other",
            "status": 2,
            "duration_ms": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
        },
    )
    calls = await ai_llm_call_repository.list_by_trace(db, "tr-llm")
    assert [c.seq for c in calls] == [1, 2]  # 按 seq 正序回放调用链路
    assert calls[0].model == "gpt-x"
    assert not any(c.model == "other" for c in calls)
