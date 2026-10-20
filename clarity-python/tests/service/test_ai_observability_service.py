"""AI 可观测性查询服务测试：总览/检索/详情归属/消耗聚合/趋势/导出/会话时间线"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_artifact import SysAiArtifact
from app.models.entity.sys_ai_billing import SysAiBilling
from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_message import SysAiMessage
from app.repository.ai_agent_thought_repository import ai_agent_thought_repository
from app.repository.ai_llm_call_repository import ai_llm_call_repository
from app.repository.ai_trace_repository import ai_trace_repository
from app.service.ai_observability_service import ai_observability_service

pytestmark = pytest.mark.requires_db

USER_A = 77
USER_B = 88


def _body_bytes(chunks) -> bytes:
    """StreamingResponse.body_iterator 声明为 Content（str|bytes|memoryview）流，
    导出实现实际产出 bytes；统一归一为 bytes 后再解码断言。"""
    parts: list[bytes] = []
    for chunk in chunks:
        if isinstance(chunk, bytes):
            parts.append(chunk)
        elif isinstance(chunk, memoryview):
            parts.append(bytes(chunk))
        else:
            parts.append(chunk.encode("utf-8"))
    return b"".join(parts)


async def _seed(db):
    conv_a = SysAiConversation(user_id=USER_A, title="a")
    conv_b = SysAiConversation(user_id=USER_B, title="b")
    db.add_all([conv_a, conv_b])
    await db.flush()

    # conv_a 会话消息：user → assistant（assistant 消息即 t1 关联的回复消息）
    msg_user = SysAiMessage(conversation_id=conv_a.id, role="user", content="你好", status=2)
    msg_assistant = SysAiMessage(
        conversation_id=conv_a.id,
        role="assistant",
        content="你好，有什么可以帮你？",
        status=2,
        model="m1",
    )
    db.add_all([msg_user, msg_assistant])
    await db.flush()

    # t1: conv_a 成功 / m1 / a1 / 含记忆构成；t2: conv_a 失败(配额拒绝) / m1
    # t3: conv_b 中断 / m2 / a2 / 高步数；t4: conv_b 超时 / m2 / 含失败的工具调用
    traces = [
        {
            "trace_id": "t1",
            "conversation_id": conv_a.id,
            "message_id": msg_assistant.id,
            "agent_code": "a1",
            "model": "m1",
            "status": 1,
            "duration_ms": 100,
            "first_token_ms": 50,
            "llm_call_count": 2,
            "total_tokens": 100,
            "prompt_tokens": 60,
            "completion_tokens": 40,
            "cached_tokens": 10,
            "step_count": 2,
            "context_snapshot": {
                "items": [
                    {"type": "system", "tokens": 30},
                    {"type": "memory", "count": 2, "tokens": 10},
                ]
            },
        },
        {
            "trace_id": "t2",
            "conversation_id": conv_a.id,
            "model": "m1",
            "status": 2,
            "error_type": "quota",
            "duration_ms": 200,
            "llm_call_count": 0,
            "total_tokens": 50,
            "step_count": 1,
            "first_token_ms": 999,
            "error_detail": {"message": "配额不足", "stack": "Traceback (latest call last)"},
        },
        {
            "trace_id": "t3",
            "conversation_id": conv_b.id,
            "agent_code": "a2",
            "model": "m2",
            "status": 3,
            "error_type": "confirm",
            "duration_ms": 300,
            "llm_call_count": 3,
            "step_count": 45,
        },
        {
            "trace_id": "t4",
            "conversation_id": conv_b.id,
            "model": "m2",
            "status": 4,
            "duration_ms": 400,
            "first_token_ms": 100,
            "llm_call_count": 1,
            "step_count": 1,
        },
    ]
    for values in traces:
        await ai_trace_repository.insert_idempotent(db, values)

    for seq, step in ((1, 1), (2, 2)):
        await ai_llm_call_repository.insert_idempotent(
            db,
            {
                "trace_id": "t1",
                "seq": seq,
                "step_position": step,
                "model": "m1",
                "status": 1,
                "duration_ms": 10 * seq,
                "prompt_tokens": 30,
                "completion_tokens": 20,
                "attempts": [
                    {
                        "provider_id": 1,
                        "key_id": 2,
                        "model": "m1",
                        "status": 1,
                        "error_code": None,
                        "latency_ms": 9,
                    }
                ],
            },
        )
    # t4 第 1 轮调用发起工具调用但调用失败（高风险工具调用口径）
    await ai_llm_call_repository.insert_idempotent(
        db,
        {
            "trace_id": "t4",
            "seq": 1,
            "step_position": 1,
            "model": "m2",
            "status": 2,
            "error_type": "TimeoutError",
            "duration_ms": 5000,
            "prompt_tokens": 30,
            "tool_call": {
                "has_tool_call": True,
                "tools": [{"name": "kb_search", "arguments": "{}"}],
            },
        },
    )
    # t1 关联 assistant 消息的推理步骤（position 升序回放）
    await ai_agent_thought_repository.create_thought(
        db,
        message_id=msg_assistant.id,
        conversation_id=conv_a.id,
        position=1,
        thought="用户在打招呼，直接回复",
        status=1,
        latency_ms=10,
    )
    await ai_agent_thought_repository.create_thought(
        db,
        message_id=msg_assistant.id,
        conversation_id=conv_a.id,
        position=2,
        thought="查询知识库",
        tool="kb_search",
        tool_input={"query": "你好"},
        observation="命中1条",
        status=1,
        latency_ms=20,
    )
    # t1 关联计费（request_id=trace_id 关联口径）与中间产物
    db.add_all(
        [
            SysAiBilling(
                user_id=USER_A,
                message_id=msg_assistant.id,
                model="m1",
                bill_type="chat",
                request_id="t1",
                input_tokens=60,
                output_tokens=40,
                credits=5,
            ),
            SysAiArtifact(
                conversation_id=conv_a.id,
                message_id=msg_assistant.id,
                type="image",
                summary={"name": "dehaze-demo"},
            ),
        ]
    )
    await db.flush()
    return conv_a, conv_b, msg_assistant.id


async def test_summary_counts(db):
    await _seed(db)
    result = await ai_observability_service.summary(db)
    assert result.total == 4
    assert result.success_count == 1
    assert result.failed_count == 1
    assert result.interrupted_count == 1
    assert result.timeout_count == 1
    assert result.quota_rejected == 1  # t2 error_type=quota
    # t3 step_count=45 + t4 存在失败的工具调用
    assert result.high_risk_calls == 2


async def test_summary_empty(db):
    result = await ai_observability_service.summary(db)
    assert result.total == 0
    assert result.quota_rejected == 0
    assert result.high_risk_calls == 0


async def test_list_traces_filters(db):
    await _seed(db)
    from app.models.schema.ai_observability import TracePageQuery

    by_model = await ai_observability_service.list_traces(
        db, TracePageQuery(model="m1", pageSize=10)
    )
    assert by_model.total == 2
    assert {t.trace_id for t in by_model.list} == {"t1", "t2"}

    by_agent = await ai_observability_service.list_traces(
        db, TracePageQuery(agentCode="a1", pageSize=10)
    )
    assert by_agent.total == 1

    by_user = await ai_observability_service.list_traces(
        db, TracePageQuery(userId=USER_B, pageSize=10)
    )
    assert by_user.total == 2
    assert {t.trace_id for t in by_user.list} == {"t3", "t4"}

    by_status = await ai_observability_service.list_traces(
        db, TracePageQuery(status=1, pageSize=10)
    )
    assert by_status.total == 1


async def test_list_traces_error_type_filter(db):
    await _seed(db)
    from app.models.schema.ai_observability import TracePageQuery

    result = await ai_observability_service.list_traces(
        db, TracePageQuery(errorType="quota", pageSize=10)
    )
    assert {t.trace_id for t in result.list} == {"t2"}

    miss = await ai_observability_service.list_traces(
        db, TracePageQuery(errorType="quota_exceeded", pageSize=10)
    )
    assert miss.total == 0


async def test_list_traces_keyword_matches_trace_id_and_title(db):
    await _seed(db)
    from app.models.schema.ai_observability import TracePageQuery

    # 匹配 trace_id
    by_trace = await ai_observability_service.list_traces(
        db, TracePageQuery(keyword="t3", pageSize=10)
    )
    assert {t.trace_id for t in by_trace.list} == {"t3"}
    # 匹配会话标题（conv_a 标题 "a"，无 trace_id 含 "a"）
    by_title = await ai_observability_service.list_traces(
        db, TracePageQuery(keyword="a", pageSize=10)
    )
    assert {t.trace_id for t in by_title.list} == {"t1", "t2"}
    # keyword 与 userId 组合不产生重复 join
    combined = await ai_observability_service.list_traces(
        db, TracePageQuery(keyword="a", userId=USER_A, pageSize=10)
    )
    assert {t.trace_id for t in combined.list} == {"t1", "t2"}


async def test_list_traces_capability_filter(db):
    await _seed(db)
    from app.models.schema.ai_observability import TracePageQuery

    by_memory = await ai_observability_service.list_traces(
        db, TracePageQuery(capability="memory", pageSize=10)
    )
    assert {t.trace_id for t in by_memory.list} == {"t1"}

    no_kb = await ai_observability_service.list_traces(
        db, TracePageQuery(capability="kb", pageSize=10)
    )
    assert no_kb.total == 0


async def test_get_trace_detail_with_llm_calls(db):
    await _seed(db)
    detail = await ai_observability_service.get_trace(db, "t1", USER_A, admin=False)
    assert detail.context_snapshot == {
        "items": [{"type": "system", "tokens": 30}, {"type": "memory", "count": 2, "tokens": 10}]
    }
    assert [c.seq for c in detail.llm_calls] == [1, 2]
    assert detail.llm_calls[0].model == "m1"
    # 物理调用尝试明细随调用明细透出（B1 审计还原）
    assert detail.llm_calls[0].attempts == [
        {
            "provider_id": 1,
            "key_id": 2,
            "model": "m1",
            "status": 1,
            "error_code": None,
            "latency_ms": 9,
        }
    ]
    # 推理步骤回放：按 position 正序，含思考/工具/观察
    assert [t.position for t in detail.thoughts] == [1, 2]
    assert detail.thoughts[0].thought == "用户在打招呼，直接回复"
    assert detail.thoughts[1].tool == "kb_search"
    assert detail.thoughts[1].observation == "命中1条"
    # 会话消息回放：user 在前 assistant 在后
    assert [m.role for m in detail.messages] == ["user", "assistant"]
    assert detail.messages[0].content == "你好"
    assert detail.messages[1].content == "你好，有什么可以帮你？"


async def test_get_trace_detail_billing_and_artifacts(db):
    await _seed(db)
    detail = await ai_observability_service.get_trace(db, "t1", USER_A, admin=False)
    # 计费按 request_id=trace_id 精确关联
    assert len(detail.billing) == 1
    billing = detail.billing[0]
    assert billing.request_id == "t1"
    assert billing.bill_type == "chat"
    assert billing.credits == 5
    assert billing.input_tokens == 60
    assert billing.output_tokens == 40
    # 产物按 message_id 关联，summary 透出
    assert len(detail.artifacts) == 1
    artifact = detail.artifacts[0]
    assert artifact.type == "image"
    assert artifact.summary == {"name": "dehaze-demo"}


async def test_get_trace_billing_fallback_by_message(db):
    """无 request_id 命中时按 message_id 回退关联（补记/兼容场景）"""
    conv_a, _, msg_id = await _seed(db)

    db.add(
        SysAiBilling(user_id=USER_A, message_id=msg_id, model="m1", bill_type="tool_llm", credits=2)
    )
    await ai_trace_repository.insert_idempotent(
        db,
        {"trace_id": "t5", "conversation_id": conv_a.id, "message_id": msg_id},
    )
    await db.flush()
    detail = await ai_observability_service.get_trace(db, "t5", USER_A, admin=False)
    # request_id 无命中 → 回退 message_id，命中该消息全部计费记录
    assert {b.bill_type for b in detail.billing} == {"chat", "tool_llm"}
    # request_id 命中优先：不混入无 request_id 记录
    detail_t1 = await ai_observability_service.get_trace(db, "t1", USER_A, admin=False)
    assert len(detail_t1.billing) == 1


async def test_get_trace_detail_bypass_trace_without_message(db):
    """旁路过程链（trace_type=summary，message_id=None）详情可查：
    thoughts/billing/artifacts 按 message_id 关联，缺失时为空，消息回放仍可用"""
    conv_a, _, _ = await _seed(db)
    await ai_trace_repository.insert_idempotent(
        db,
        {
            "trace_id": "t6",
            "conversation_id": conv_a.id,
            "message_id": None,
            "model": "m1",
            "status": 1,
            "trace_type": "summary",
            "duration_ms": 80,
            "llm_call_count": 1,
            "total_tokens": 30,
            "prompt_tokens": 20,
            "completion_tokens": 10,
        },
    )
    await db.flush()
    detail = await ai_observability_service.get_trace(db, "t6", USER_A, admin=False)
    assert detail.message_id is None
    assert detail.thoughts == []
    assert detail.billing == []
    assert detail.artifacts == []
    assert detail.llm_calls == []
    assert [m.role for m in detail.messages] == ["user", "assistant"]


async def test_get_trace_owner_vs_other_user(db):
    conv_a, _, _ = await _seed(db)
    # 归属用户可查（普通用户身份），失败链路异常详情（消息+堆栈）透出
    detail = await ai_observability_service.get_trace(db, "t1", USER_A, admin=False)
    assert detail.trace_id == "t1"
    failed = await ai_observability_service.get_trace(db, "t2", USER_A, admin=False)
    assert failed.error_detail == {"message": "配额不足", "stack": "Traceback (latest call last)"}
    # 非归属用户 A0401，不暴露存在性
    with pytest.raises(BusinessException) as exc:
        await ai_observability_service.get_trace(db, "t1", USER_B, admin=False)
    assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND
    # 管理员全量可查
    admin_detail = await ai_observability_service.get_trace(db, "t1", USER_B, admin=True)
    assert admin_detail.trace_id == "t1"
    assert conv_a is not None


async def test_get_trace_not_found(db):
    with pytest.raises(BusinessException) as exc:
        await ai_observability_service.get_trace(db, "nope", USER_A, admin=True)
    assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND


async def test_costs_by_model_and_user(db):
    await _seed(db)
    from app.models.schema.ai_observability import CostsQuery

    by_model = await ai_observability_service.costs(db, CostsQuery(dimension="model", pageSize=10))
    assert by_model.total == 2
    m1 = next(i for i in by_model.items if i.model == "m1")
    assert m1.trace_count == 2
    assert m1.total_tokens == 150
    assert m1.prompt_tokens == 60
    assert m1.completion_tokens == 40
    assert m1.cached_tokens == 10
    # 按日趋势聚合全部维度
    assert len(by_model.trend) == 1
    assert by_model.trend[0].trace_count == 4
    assert by_model.trend[0].total_tokens == 150

    by_user = await ai_observability_service.costs(db, CostsQuery(dimension="user", pageSize=10))
    assert by_user.total == 2
    mine = next(i for i in by_user.items if i.user_id == USER_A)
    assert mine.trace_count == 2
    assert mine.total_tokens == 150


async def test_trends_success_rate_and_latency(db):
    await _seed(db)
    from app.models.schema.ai_observability import TrendsQuery

    items = await ai_observability_service.trends(db, TrendsQuery(dimension="model"))
    m1 = next(i for i in items if i.model == "m1")
    assert m1.call_count == 2
    assert m1.success_count == 1
    assert m1.success_rate == 50.0
    # 首 Token 延迟成功调用口径：t1=50，失败链路 t2 虽有 first_token_ms=999 也不计入
    assert m1.avg_first_token_ms == 50.0
    assert m1.avg_duration_ms == 150.0  # (100+200)/2


async def test_export_traces_csv(db):
    await _seed(db)
    from app.models.schema.ai_observability import TracePageQuery

    resp = await ai_observability_service.export_traces(db, TracePageQuery(pageSize=10))
    assert resp.media_type == "text/csv"
    assert "ai_traces.csv" in resp.headers["content-disposition"]
    chunks = [chunk async for chunk in resp.body_iterator]
    text = _body_bytes(chunks).decode("utf-8")
    assert text.startswith("\ufefftrace_id,")
    for trace_id in ("t1", "t2", "t3", "t4"):
        assert trace_id in text


# ---------------------------------------------------------------------------
# 会话时间线（审计级重构设计 §4.2）
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 9, 13, 10, 0, 0)


def _dt(seconds: float) -> datetime:
    return _T0 + timedelta(seconds=seconds)


async def _seed_timeline(db):
    """两轮会话（激活分支链）+ 轮2主对话 trace（resume 两条并列）+ 旁路 trace。

    事件 ts 全部经 create_time-duration 近似锚定（老数据无 start_time 的口径）：
    轮2 tA 内事件序 = input(0s) → context(0s,同刻按 kind 序) → llm#1(3s)
    → tool_exec(15s) → llm#2(20s) → billing(29s)。
    """
    conv = SysAiConversation(user_id=USER_A, title="审计会话")
    db.add(conv)
    await db.flush()

    u1 = SysAiMessage(
        conversation_id=conv.id, role="user", content="第一问", status=2, create_time=_dt(0)
    )
    db.add(u1)
    await db.flush()
    a1 = SysAiMessage(
        conversation_id=conv.id,
        parent_message_id=u1.id,
        role="assistant",
        content="第一答",
        status=2,
        model="m1",
        input_tokens=10,
        output_tokens=5,
        create_time=_dt(5),
    )
    db.add(a1)
    await db.flush()
    u2 = SysAiMessage(
        conversation_id=conv.id,
        parent_message_id=a1.id,
        role="user",
        content="第二问",
        status=2,
        create_time=_dt(60),
    )
    db.add(u2)
    await db.flush()
    a2 = SysAiMessage(
        conversation_id=conv.id,
        parent_message_id=u2.id,
        role="assistant",
        content="第二答",
        status=2,
        model="m1",
        input_tokens=30,
        output_tokens=20,
        create_time=_dt(90),
    )
    db.add(a2)
    await db.flush()
    conv.current_branch_message_id = a2.id

    traces = [
        # 轮1 主对话
        {
            "trace_id": "r1",
            "conversation_id": conv.id,
            "message_id": a1.id,
            "model": "m1",
            "status": 1,
            "duration_ms": 5000,
            "llm_call_count": 1,
            "total_tokens": 15,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "create_time": _dt(5),
        },
        # 轮2 主对话（context 近似 = create_time - duration = 轮2触发时刻）
        {
            "trace_id": "tA",
            "conversation_id": conv.id,
            "message_id": a2.id,
            "model": "m1",
            "status": 1,
            "duration_ms": 30000,
            "llm_call_count": 2,
            "total_tokens": 50,
            "prompt_tokens": 30,
            "completion_tokens": 20,
            "create_time": _dt(90),
            "context_snapshot": {
                "items": [{"type": "system"}],
                "events": [{"event": "summarize", "tokens": 100}],
            },
        },
        # 轮2 resume 续流（同一助手消息并列第二条主对话 trace）
        {
            "trace_id": "tB",
            "conversation_id": conv.id,
            "message_id": a2.id,
            "model": "m1",
            "status": 1,
            "duration_ms": 1000,
            "llm_call_count": 0,
            "total_tokens": 0,
            "create_time": _dt(120),
        },
        # 旁路：记忆提取（无 message_id，触发时点 140s → 挂轮2尾部）
        {
            "trace_id": "tM",
            "conversation_id": conv.id,
            "message_id": None,
            "trace_type": "memory_extraction",
            "model": "m1",
            "status": 1,
            "duration_ms": 800,
            "llm_call_count": 1,
            "total_tokens": 10,
            "create_time": _dt(140),
        },
        # 旁路：建议推荐（有 message_id，挂触发轮次尾部）
        {
            "trace_id": "tS",
            "conversation_id": conv.id,
            "message_id": a2.id,
            "trace_type": "suggestion",
            "model": "m1",
            "status": 1,
            "duration_ms": 600,
            "llm_call_count": 1,
            "total_tokens": 8,
            "create_time": _dt(150),
        },
    ]
    for values in traces:
        await ai_trace_repository.insert_idempotent(db, values)

    # tA 的两次 LLM 调用：#1 新数据（含 start_time/raw 报文，ts=start_time=63s），
    # #2 老数据（无 raw，ts 经 create_time-duration 近似为 80s）
    await ai_llm_call_repository.insert_idempotent(
        db,
        {
            "trace_id": "tA",
            "seq": 1,
            "step_position": 1,
            "model": "m1",
            "status": 1,
            "duration_ms": 2000,
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "cached_tokens": 0,
            "input_snapshot": {
                "messages": {
                    "counts": {"user": 1, "assistant": 1},
                    "tokens": 30,
                    "items": [{"role": "user", "content": "第二问"}],
                },
                "tool_count": 1,
                "tools": [{"name": "kb_search"}],
            },
            "output_snapshot": {"text": "第一步"},
            "start_time": _dt(63),
            "raw_request": {"model": "m1", "messages": [{"role": "user", "content": "第二问"}]},
            "raw_response": {"choices": [{"finish_reason": "stop"}]},
            "create_time": _dt(65),
        },
    )
    await ai_llm_call_repository.insert_idempotent(
        db,
        {
            "trace_id": "tA",
            "seq": 2,
            "step_position": 2,
            "model": "m1",
            "status": 1,
            "duration_ms": 5000,
            "prompt_tokens": 30,
            "completion_tokens": 20,
            "cached_tokens": 5,
            "tool_call": {"has_tool_call": False},
            "input_snapshot": {
                "messages": {
                    "counts": {"user": 2},
                    "tokens": 40,
                    "items": [{"role": "user", "content": "第二问"}],
                }
            },
            "create_time": _dt(85),
        },
    )
    await ai_agent_thought_repository.create_thought(
        db,
        message_id=a2.id,
        conversation_id=conv.id,
        position=1,
        thought="调用搜索",
        tool="kb_search",
        tool_input={"query": "第二问"},
        observation="命中2条",
        status=1,
        latency_ms=800,
        create_time=_dt(75),
    )
    db.add_all(
        [
            SysAiBilling(
                user_id=USER_A,
                conversation_id=conv.id,
                model="m1",
                bill_type="chat",
                request_id="tA",
                input_tokens=30,
                output_tokens=20,
                cached_input_tokens=5,
                credits=3,
                create_time=_dt(89),
            ),
            SysAiBilling(
                user_id=USER_A,
                conversation_id=conv.id,
                model="m1",
                bill_type="tool_llm",
                message_id=a2.id,
                input_tokens=5,
                output_tokens=1,
                credits=1,
                create_time=_dt(88),
            ),
        ]
    )
    await db.flush()
    return conv


async def test_timeline_rounds_split_by_branch_chain(db):
    conv = await _seed_timeline(db)
    result = await ai_observability_service.get_conversation_timeline(
        db, conv.id, USER_A, admin=False
    )
    assert result.conversation.id == conv.id
    assert result.conversation.title == "审计会话"
    assert result.conversation.user_id == USER_A
    assert len(result.rounds) == 2
    r1, r2 = result.rounds
    assert r1.user_message is not None
    assert r1.assistant_message is not None
    assert r2.user_message is not None
    assert r2.assistant_message is not None
    assert r1.user_message.content == "第一问"
    assert r1.assistant_message.content == "第一答"
    assert r2.user_message.content == "第二问"
    assert r2.assistant_message.content == "第二答"
    assert r2.assistant_message.input_tokens == 30


async def test_timeline_events_interweaving_order(db):
    conv = await _seed_timeline(db)
    result = await ai_observability_service.get_conversation_timeline(
        db, conv.id, USER_A, admin=False
    )
    main = result.rounds[1].traces[0]
    assert main.trace_id == "tA"
    kinds = [e.kind for e in main.events]
    # input 与 context 同刻（近似锚点），按 kind 优先级 input 在前；
    # llm#2 无 start_time（ts NULL）沉底，排在 billing 之后
    assert kinds == [
        "input",
        "context",
        "system_event",
        "llm_call",
        "tool_exec",
        "billing",
        "llm_call",
    ]
    # 同刻 llm_call 按业务序 seq 排序；调用字段完整透出
    llm_events = [e for e in main.events if e.kind == "llm_call"]
    assert [e.seq for e in llm_events] == [1, 2]
    assert llm_events[0].model == "m1"
    assert llm_events[0].duration_ms == 2000
    assert llm_events[0].prompt_tokens == 20
    assert llm_events[0].completion_tokens == 10
    assert llm_events[0].ts == _dt(63)  # ts 即 start_time，无近似
    # summary 恒为纯计数单形状（messages/tools 全文唯一通道走 rawRequest）
    assert llm_events[0].raw_request == {
        "model": "m1",
        "messages": [{"role": "user", "content": "第二问"}],
    }
    assert llm_events[0].raw_response == {"choices": [{"finish_reason": "stop"}]}
    assert llm_events[0].summary is not None
    assert llm_events[0].summary["inputSnapshot"] == {
        "messages": {"counts": {"user": 1, "assistant": 1}, "tokens": 30},
        "tool_count": 1,
    }
    assert llm_events[0].summary["outputSnapshot"] == {"text": "第一步"}
    # #2 raw/start_time 为 NULL：无原始报文（前端空态），summary 仍为计数字段
    assert llm_events[1].ts is None
    assert llm_events[1].cached_tokens == 5
    assert llm_events[1].tool_call == {"has_tool_call": False}
    assert llm_events[1].raw_request is None
    assert llm_events[1].summary is not None
    assert llm_events[1].summary["inputSnapshot"] == {
        "messages": {"counts": {"user": 2}, "tokens": 40}
    }
    # tool_exec 自 thought（position/入参/观察/归属）
    tool = next(e for e in main.events if e.kind == "tool_exec")
    assert (tool.position, tool.tool) == (1, "kb_search")
    assert tool.tool_input == {"query": "第二问"}
    assert tool.observation == "命中2条"
    assert tool.latency_ms == 800
    # context 携带快照，system_event 自 context_snapshot.events
    assert main.events[1].snapshot == {
        "items": [{"type": "system"}],
        "events": [{"event": "summarize", "tokens": 100}],
    }
    system = main.events[2]
    assert system.detail is not None
    assert (system.event, system.detail["tokens"]) == ("summarize", 100)
    # billing 按请求归因，token 明细透出
    bill = next(e for e in main.events if e.kind == "billing")
    assert (bill.bill_type, bill.credits) == ("chat", 3)
    assert bill.tokens == {"input": 30, "output": 20, "cached": 5}


async def test_timeline_resume_and_bypass_attribution(db):
    conv = await _seed_timeline(db)
    result = await ai_observability_service.get_conversation_timeline(
        db, conv.id, USER_A, admin=False
    )
    r2 = result.rounds[1]
    # 主对话在前（resume 两条并列），旁路按 create_time 排后
    assert [t.trace_id for t in r2.traces] == ["tA", "tB", "tM", "tS"]
    assert [t.trace_type for t in r2.traces] == [
        "conversation",
        "conversation",
        "memory_extraction",
        "suggestion",
    ]
    # tB 无调用事件，仅 context；旁路事件不混入主调用序列
    assert [e.kind for e in r2.traces[1].events] == ["context"]
    # 轮1 仅主对话 trace
    assert [t.trace_id for t in result.rounds[0].traces] == ["r1"]
    # request_id 归因命中优先，不混入无 request_id 的 message_id 补记计费（与详情口径一致）
    tool_llm = [e for e in r2.traces[0].events if e.kind == "billing" and e.bill_type == "tool_llm"]
    assert len(tool_llm) == 0


async def test_timeline_include_raw_switch(db):
    conv = await _seed_timeline(db)
    with_raw = await ai_observability_service.get_conversation_timeline(
        db, conv.id, USER_A, admin=False, include_raw=True
    )
    without_raw = await ai_observability_service.get_conversation_timeline(
        db, conv.id, USER_A, admin=False, include_raw=False
    )
    llm_with = [e for e in with_raw.rounds[1].traces[0].events if e.kind == "llm_call"]
    llm_without = [e for e in without_raw.rounds[1].traces[0].events if e.kind == "llm_call"]
    # include=raw：raw 非空调用透传 wire 报文
    assert llm_with[0].raw_request is not None
    assert llm_with[1].raw_request is None  # 无原始报文的调用即 NULL，无兜底
    # include 非 raw：一律省略 raw
    assert all(e.raw_request is None and e.raw_response is None for e in llm_without)
    # summary 恒为纯计数单形状，与 include/raw 是否在场无关
    for events in (llm_with, llm_without):
        for e in events:
            assert e.summary is not None
            assert "items" not in e.summary["inputSnapshot"]["messages"]
    assert [e.seq for e in llm_without] == [e.seq for e in llm_with]


def test_trace_events_raw_wire_contract():
    """raw 三字段契约 + summary 单形状：inputSnapshot 恒为纯计数（messages/tools
    全文唯一通道走 rawRequest），ts 即 start_time（NULL 沉底，无近似估算）"""
    full_snapshot = {
        "messages": {
            "counts": {"user": 2},
            "tokens": 30,
            "items": [{"role": "user", "content": "消息全文"}],
        },
        "system_tokens": 100,
        "system_content": "系统提示全文",
        "tool_count": 2,
        "tools": [{"name": "kb_search"}],
        "user_id": 7,
    }
    slim_snapshot = {
        "messages": {"counts": {"user": 2}, "tokens": 30},
        "system_tokens": 100,
        "tool_count": 2,
        "user_id": 7,
    }

    def _make_call(raw_request, start_time: datetime | None = _dt(1)):
        return SimpleNamespace(
            seq=1,
            step_position=1,
            model="m1",
            status=1,
            error_type=None,
            duration_ms=100,
            first_token_ms=50,
            prompt_tokens=10,
            completion_tokens=5,
            cached_tokens=0,
            tool_call=None,
            input_snapshot=full_snapshot,
            output_snapshot={"text": "hi"},
            attempts=[],
            start_time=start_time,
            raw_request=raw_request,
            raw_response={"usage": {}},
            create_time=_dt(2),
        )

    trace = SimpleNamespace(
        trace_id="x",
        context_snapshot={},
        create_time=_dt(2),
        duration_ms=100,
    )

    # raw 在场：summary 瘦身为纯计数，raw 报文透传，ts = start_time
    events = ai_observability_service._trace_events(
        trace, [_make_call({"model": "m1"})], [], [], None, include_raw=True
    )
    llm = next(e for e in events if e.kind == "llm_call")
    assert llm.ts == _dt(1)
    assert llm.raw_request == {"model": "m1"}
    assert llm.raw_response == {"usage": {}}
    assert llm.summary is not None
    assert llm.summary["inputSnapshot"] == slim_snapshot
    assert llm.summary["outputSnapshot"] == {"text": "hi"}

    # raw 为 NULL：无原始报文，summary 仍为纯计数（无全文兜底形状）
    events = ai_observability_service._trace_events(
        trace, [_make_call(None)], [], [], None, include_raw=True
    )
    llm = next(e for e in events if e.kind == "llm_call")
    assert llm.raw_request is None
    assert llm.summary is not None
    assert llm.summary["inputSnapshot"] == slim_snapshot

    # start_time 为 NULL：ts 即 NULL（排序沉底），不做近似估算
    events = ai_observability_service._trace_events(
        trace, [_make_call({"model": "m1"}, start_time=None)], [], [], None, include_raw=True
    )
    llm = next(e for e in events if e.kind == "llm_call")
    assert llm.ts is None

    # include 非 raw：响应不含 raw，summary 形状不变
    events = ai_observability_service._trace_events(
        trace, [_make_call({"model": "m1"})], [], [], None, include_raw=False
    )
    llm = next(e for e in events if e.kind == "llm_call")
    assert llm.raw_request is None
    assert llm.raw_response is None
    assert llm.summary is not None
    assert llm.summary["inputSnapshot"] == slim_snapshot


async def test_timeline_owner_vs_other_user(db):
    conv = await _seed_timeline(db)
    result = await ai_observability_service.get_conversation_timeline(
        db, conv.id, USER_A, admin=False
    )
    assert result.conversation.id == conv.id
    # 非归属用户 A0401，不暴露存在性
    with pytest.raises(BusinessException) as exc:
        await ai_observability_service.get_conversation_timeline(db, conv.id, USER_B, admin=False)
    assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND
    # 管理员全量可查
    admin_result = await ai_observability_service.get_conversation_timeline(
        db, conv.id, USER_B, admin=True
    )
    assert admin_result.conversation.id == conv.id


async def test_timeline_conversation_not_found(db):
    with pytest.raises(BusinessException) as exc:
        await ai_observability_service.get_conversation_timeline(db, 999999, USER_A, admin=False)
    assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND


async def test_list_traces_conversation_title(db):
    await _seed(db)
    from app.models.schema.ai_observability import TracePageQuery

    result = await ai_observability_service.list_traces(
        db, TracePageQuery(userId=USER_A, pageSize=10)
    )
    assert {t.conversation_title for t in result.list} == {"a"}


async def test_export_timeline_json(db):
    conv = await _seed_timeline(db)
    resp = await ai_observability_service.export_timeline(db, conv.id, USER_A)
    assert resp.media_type == "application/json"
    assert f"conversation_{conv.id}_timeline.json" in resp.headers["content-disposition"]
    chunks = [chunk async for chunk in resp.body_iterator]
    import json

    data = json.loads(_body_bytes(chunks).decode("utf-8"))
    # 导出为 JSON 全量（camelCase 含 raw 报文），轮次与事件结构完整
    assert data["conversation"]["id"] == conv.id
    assert len(data["rounds"]) == 2
    assert [e["kind"] for e in data["rounds"][1]["traces"][0]["events"]] == [
        "input",
        "context",
        "system_event",
        "llm_call",
        "tool_exec",
        "billing",
        "llm_call",
    ]
