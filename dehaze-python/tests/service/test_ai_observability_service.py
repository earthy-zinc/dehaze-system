"""AI 可观测性查询服务测试：总览/检索/详情归属/消耗聚合/趋势/导出"""

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


async def _seed(db):
    conv_a = SysAiConversation(user_id=USER_A, title="a")
    conv_b = SysAiConversation(user_id=USER_B, title="b")
    db.add_all([conv_a, conv_b])
    await db.flush()

    # conv_a 会话消息：user → assistant（assistant 消息即 t1 关联的回复消息）
    msg_user = SysAiMessage(conversation_id=conv_a.id, role="user", content="你好", status=2)
    msg_assistant = SysAiMessage(
        conversation_id=conv_a.id, role="assistant", content="你好，有什么可以帮你？",
        status=2, model="m1",
    )
    db.add_all([msg_user, msg_assistant])
    await db.flush()

    # t1: conv_a 成功 / m1 / a1 / 含记忆构成；t2: conv_a 失败(配额拒绝) / m1
    # t3: conv_b 中断 / m2 / a2 / 高步数；t4: conv_b 超时 / m2 / 含失败的工具调用
    traces = [
        {"trace_id": "t1", "conversation_id": conv_a.id, "message_id": msg_assistant.id,
         "agent_code": "a1", "model": "m1",
         "status": 1, "duration_ms": 100, "first_token_ms": 50, "llm_call_count": 2,
         "total_tokens": 100, "prompt_tokens": 60, "completion_tokens": 40, "cached_tokens": 10,
         "step_count": 2,
         "context_snapshot": {"items": [{"type": "system", "tokens": 30},
                                        {"type": "memory", "count": 2, "tokens": 10}]}},
        {"trace_id": "t2", "conversation_id": conv_a.id, "model": "m1", "status": 2,
         "error_type": "quota", "duration_ms": 200, "llm_call_count": 0, "total_tokens": 50,
         "step_count": 1},
        {"trace_id": "t3", "conversation_id": conv_b.id, "agent_code": "a2", "model": "m2",
         "status": 3, "error_type": "confirm", "duration_ms": 300, "llm_call_count": 3,
         "step_count": 45},
        {"trace_id": "t4", "conversation_id": conv_b.id, "model": "m2", "status": 4,
         "duration_ms": 400, "first_token_ms": 100, "llm_call_count": 1, "step_count": 1},
    ]
    for values in traces:
        await ai_trace_repository.insert_idempotent(db, values)

    for seq, step in ((1, 1), (2, 2)):
        await ai_llm_call_repository.insert_idempotent(
            db,
            {"trace_id": "t1", "seq": seq, "step_position": step, "model": "m1", "status": 1,
             "duration_ms": 10 * seq, "prompt_tokens": 30, "completion_tokens": 20},
        )
    # t4 第 1 轮调用发起工具调用但调用失败（高风险工具调用口径）
    await ai_llm_call_repository.insert_idempotent(
        db,
        {"trace_id": "t4", "seq": 1, "step_position": 1, "model": "m2", "status": 2,
         "error_type": "TimeoutError", "duration_ms": 5000, "prompt_tokens": 30,
         "tool_call": {"has_tool_call": True, "tools": [{"name": "kb_search", "arguments": "{}"}]}},
    )
    # t1 关联 assistant 消息的推理步骤（position 升序回放）
    await ai_agent_thought_repository.create_thought(
        db, message_id=msg_assistant.id, conversation_id=conv_a.id, position=1,
        thought="用户在打招呼，直接回复", status=1, latency_ms=10,
    )
    await ai_agent_thought_repository.create_thought(
        db, message_id=msg_assistant.id, conversation_id=conv_a.id, position=2,
        thought="查询知识库", tool="kb_search", tool_input={"query": "你好"},
        observation="命中1条", status=1, latency_ms=20,
    )
    # t1 关联计费（request_id=trace_id 关联口径）与中间产物
    db.add_all(
        [
            SysAiBilling(
                user_id=USER_A, message_id=msg_assistant.id, model="m1", bill_type="chat",
                request_id="t1", input_tokens=60, output_tokens=40, credits=5,
            ),
            SysAiArtifact(
                conversation_id=conv_a.id, message_id=msg_assistant.id, type="image",
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
    assert billing.bill_type == "chat" and billing.credits == 5
    assert billing.input_tokens == 60 and billing.output_tokens == 40
    # 产物按 message_id 关联，summary 透出
    assert len(detail.artifacts) == 1
    artifact = detail.artifacts[0]
    assert artifact.type == "image"
    assert artifact.summary == {"name": "dehaze-demo"}


async def test_get_trace_billing_fallback_by_message(db):
    """无 request_id 命中时按 message_id 回退关联（补记/兼容场景）"""
    conv_a, _, msg_id = await _seed(db)
    from sqlalchemy import select

    db.add(
        SysAiBilling(
            user_id=USER_A, message_id=msg_id, model="m1", bill_type="tool_llm", credits=2
        )
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
    # 归属用户可查（普通用户身份）
    detail = await ai_observability_service.get_trace(db, "t1", USER_A, admin=False)
    assert detail.trace_id == "t1"
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
    # 首Token延迟仅成功调用有值：t1=50，t2 无
    assert m1.avg_first_token_ms == 50.0
    assert m1.avg_duration_ms == 150.0  # (100+200)/2


async def test_export_traces_csv(db):
    await _seed(db)
    from app.models.schema.ai_observability import TracePageQuery

    resp = await ai_observability_service.export_traces(db, TracePageQuery(pageSize=10))
    assert resp.media_type == "text/csv"
    assert "ai_traces.csv" in resp.headers["content-disposition"]
    chunks = [chunk async for chunk in resp.body_iterator]
    text = b"".join(chunks).decode("utf-8")
    assert text.startswith("\ufefftrace_id,")
    for trace_id in ("t1", "t2", "t3", "t4"):
        assert trace_id in text
