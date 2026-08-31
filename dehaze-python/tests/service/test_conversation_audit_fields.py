"""管理端会话审计字段（用户名/消耗汇总/异常标注）与搜索命中消息定位单测"""

from datetime import datetime

import pytest

pytestmark = pytest.mark.requires_db

from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_billing import SysAiBilling
from app.models.entity.sys_ai_billing_anomaly import SysAiBillingAnomaly
from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_message import SysAiMessage
from app.models.entity.sys_user import SysUser
from app.repository.ai_conversation_repository import ai_conversation_repository
from app.repository.ai_llm_call_repository import ai_llm_call_repository
from app.repository.ai_message_repository import ai_message_repository
from app.repository.ai_trace_repository import ai_trace_repository
from app.repository.user_repository import user_repository
from app.service import ai_conversation_service as m
from app.service.ai_conversation_service import AiConversationService


def _conv(cid, uid, title):
    return SysAiConversation(
        id=cid,
        user_id=uid,
        title=title,
        model="qwen3-0.6b",
        message_count=0,
        status=1,
        title_source="auto",
    )


def _msg(mid, cid, role="assistant", content="", status=2, deleted=0):
    return SysAiMessage(
        id=mid,
        conversation_id=cid,
        role=role,
        content=content,
        status=status,
        model="qwen3-0.6b",
        deleted=deleted,
    )


def _billing(cid, uid, *, input_tokens=0, output_tokens=0, credits=0, bill_type="chat"):
    return SysAiBilling(
        user_id=uid,
        conversation_id=cid,
        model="qwen3-0.6b",
        bill_type=bill_type,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        credits=credits,
    )


def _quota_anomaly(uid, billing_id):
    return SysAiBillingAnomaly(
        user_id=uid,
        billing_id=billing_id,
        anomaly_type="consecutive_quota_fail",
        detail="连续配额不足",
        status=0,
        trigger_at=datetime.now(),
    )


def _service(**kw):
    return AiConversationService(
        ai_conversation_repository=ai_conversation_repository,
        ai_message_repository=ai_message_repository,
        user_repository=user_repository,
        **kw,
    )


async def _seed_users(db):
    zhangsan = SysUser(username="zhangsan", nickname="张三", password="x", status=1)
    lisi = SysUser(username="lisi", password="x", status=1)
    db.add_all([zhangsan, lisi])
    await db.flush()
    return zhangsan.id, lisi.id


async def _seed_two_conversations(db):
    """会话 1（用户A）：含计费、失败消息、命中关键词消息；会话 2（用户B）：无计费无异常"""
    user_a, user_b = await _seed_users(db)
    db.add_all([_conv(1, user_a, "雾霾成因分析"), _conv(2, user_b, "算法推荐")])
    db.add_all(
        [
            _msg(11, 1, "user", "帮我分析雾霾成因"),
            _msg(12, 1, "assistant", "RIDCP 算法可处理该场景", status=3),
            _msg(21, 2, "user", "推荐一个算法"),
        ]
    )
    db.add_all(
        [
            _billing(1, user_a, input_tokens=100, output_tokens=50, credits=30),
            _billing(1, user_a, input_tokens=10, output_tokens=5, credits=7, bill_type="tool_llm"),
        ]
    )
    await db.flush()
    return user_a, user_b


class TestAdminListAuditFields:
    async def test_admin_list_aggregates_username_and_consumption(self, db):
        await _seed_two_conversations(db)
        result = await _service().list_conversations(db, 0, 1, 10, view="admin")
        by_id = {c.id: c for c in result.list}

        assert by_id[1].user_name == "张三"
        assert by_id[1].token_consumed == 165
        assert by_id[1].credits_consumed == 37

    async def test_admin_list_defaults_zero_without_billing(self, db):
        await _seed_two_conversations(db)
        result = await _service().list_conversations(db, 0, 1, 10, view="admin")
        by_id = {c.id: c for c in result.list}

        assert by_id[2].token_consumed == 0
        assert by_id[2].credits_consumed == 0
        assert by_id[2].anomaly_type is None

    async def test_admin_list_username_falls_back_to_account(self, db):
        await _seed_two_conversations(db)
        result = await _service().list_conversations(db, 0, 1, 10, view="admin")
        assert {c.id: c for c in result.list}[2].user_name == "lisi"

    async def test_admin_list_with_keyword_stays_in_audit_scope(self, db):
        """view=admin 带关键词走 DB 标题 like（不落用户视角 ES 检索），且仍带审计字段"""
        user_a, _ = await _seed_two_conversations(db)
        result = await _service().list_conversations(
            db, 0, 1, 10, keyword="雾霾", view="admin"
        )

        assert [c.id for c in result.list] == [1]
        assert result.list[0].user_id == user_a
        assert result.list[0].token_consumed == 165
        assert result.list[0].anomaly_type == "failed"

    async def test_admin_list_keyword_miss_returns_empty(self, db):
        await _seed_two_conversations(db)
        result = await _service().list_conversations(db, 0, 1, 10, keyword="不存在", view="admin")

        assert result.list == []
        assert result.total == 0

    async def test_user_list_does_not_aggregate(self, db):
        user_a, _ = await _seed_two_conversations(db)
        result = await _service().list_conversations(db, user_a, 1, 10)
        conv = {c.id: c for c in result.list}[1]

        assert conv.user_name is None
        assert conv.token_consumed is None
        assert conv.credits_consumed is None
        assert conv.anomaly_type is None


class TestAnomalyLabel:
    async def test_failed_message_marks_failed(self, db):
        await _seed_two_conversations(db)
        result = await _service().list_conversations(db, 0, 1, 10, view="admin")
        conv = {c.id: c for c in result.list}[1]

        assert conv.anomaly_type == "failed"
        assert conv.anomaly_label == "存在失败消息"

    async def test_canceled_message_marks_canceled(self, db):
        user_a, _ = await _seed_users(db)
        db.add(_conv(1, user_a, "会话"))
        db.add_all(
            [
                _msg(11, 1, "user", "提问"),
                _msg(12, 1, "assistant", "未完成", status=4),
            ]
        )
        await db.flush()

        result = await _service().list_conversations(db, 0, 1, 10, view="admin")
        conv = {c.id: c for c in result.list}[1]

        assert conv.anomaly_type == "canceled"
        assert conv.anomaly_label == "存在已取消消息"

    async def test_quota_anomaly_marks_quota(self, db):
        user_a, _ = await _seed_users(db)
        db.add(_conv(1, user_a, "会话"))
        billing = _billing(1, user_a, input_tokens=10, credits=1)
        db.add(billing)
        await db.flush()
        db.add(_quota_anomaly(user_a, billing.id))
        await db.flush()

        result = await _service().list_conversations(db, 0, 1, 10, view="admin")
        conv = {c.id: c for c in result.list}[1]

        assert conv.anomaly_type == "quota"
        assert conv.anomaly_label == "配额不足中断"

    async def test_failed_takes_priority_over_canceled_and_quota(self, db):
        user_a, _ = await _seed_users(db)
        db.add(_conv(1, user_a, "会话"))
        billing = _billing(1, user_a, credits=1)
        db.add(billing)
        db.add_all(
            [
                _msg(11, 1, "assistant", "失败回复", status=3),
                _msg(12, 1, "assistant", "取消回复", status=4),
            ]
        )
        await db.flush()
        db.add(_quota_anomaly(user_a, billing.id))
        await db.flush()

        result = await _service().list_conversations(db, 0, 1, 10, view="admin")
        assert {c.id: c for c in result.list}[1].anomaly_type == "failed"

    async def test_risky_tool_call_marks_risky_tool(self, db):
        """存在失败的工具调用（llm_call tool_call 非空且未成功）的会话标注 risky_tool"""
        user_a, _ = await _seed_users(db)
        db.add(_conv(1, user_a, "会话"))
        await db.flush()
        await ai_trace_repository.insert_idempotent(
            db, {"trace_id": "tr-risky", "conversation_id": 1, "status": 4, "duration_ms": 100}
        )
        await ai_llm_call_repository.insert_idempotent(
            db,
            {"trace_id": "tr-risky", "seq": 1, "status": 2, "error_type": "TimeoutError",
             "duration_ms": 5000, "tool_call": {"has_tool_call": True,
                                                "tools": [{"name": "kb_search"}]}},
        )

        result = await _service().list_conversations(db, 0, 1, 10, view="admin")
        conv = {c.id: c for c in result.list}[1]

        assert conv.anomaly_type == "risky_tool"
        assert conv.anomaly_label == "存在高风险工具调用"

    async def test_successful_tool_call_not_marked(self, db):
        """成功的工具调用不触发 risky_tool 标注"""
        user_a, _ = await _seed_users(db)
        db.add(_conv(1, user_a, "会话"))
        await db.flush()
        await ai_trace_repository.insert_idempotent(
            db, {"trace_id": "tr-ok", "conversation_id": 1, "status": 1, "duration_ms": 100}
        )
        await ai_llm_call_repository.insert_idempotent(
            db,
            {"trace_id": "tr-ok", "seq": 1, "status": 1, "duration_ms": 500,
             "tool_call": {"has_tool_call": True, "tools": [{"name": "kb_search"}]}},
        )

        result = await _service().list_conversations(db, 0, 1, 10, view="admin")
        assert {c.id: c for c in result.list}[1].anomaly_type is None

    async def test_failed_takes_priority_over_risky_tool(self, db):
        user_a, _ = await _seed_users(db)
        db.add(_conv(1, user_a, "会话"))
        db.add(_msg(11, 1, "assistant", "失败回复", status=3))
        await db.flush()
        await ai_trace_repository.insert_idempotent(
            db, {"trace_id": "tr-risky", "conversation_id": 1, "status": 4, "duration_ms": 100}
        )
        await ai_llm_call_repository.insert_idempotent(
            db,
            {"trace_id": "tr-risky", "seq": 1, "status": 2, "error_type": "TimeoutError",
             "duration_ms": 5000, "tool_call": {"has_tool_call": True, "tools": []}},
        )

        result = await _service().list_conversations(db, 0, 1, 10, view="admin")
        assert {c.id: c for c in result.list}[1].anomaly_type == "failed"

    async def test_other_anomaly_type_not_marked(self, db):
        """仅 consecutive_quota_fail 属配额类异常，single_high 不标注为 quota"""
        user_a, _ = await _seed_users(db)
        db.add(_conv(1, user_a, "会话"))
        billing = _billing(1, user_a, credits=1)
        db.add(billing)
        await db.flush()
        db.add(
            SysAiBillingAnomaly(
                user_id=user_a,
                billing_id=billing.id,
                anomaly_type="single_high",
                detail="单笔高额",
                status=0,
                trigger_at=datetime.now(),
            )
        )
        await db.flush()

        result = await _service().list_conversations(db, 0, 1, 10, view="admin")
        assert {c.id: c for c in result.list}[1].anomaly_type is None


class TestMatchedMessageId:
    async def test_keyword_in_message_content_backfills_latest_match(self, db, monkeypatch):
        user_a, _ = await _seed_two_conversations(db)
        db.add(_msg(13, 1, "assistant", "再次提到 RIDCP 算法", status=2))
        await db.flush()

        async def search(user_id, query, *, status, page, size):
            return [1], 1

        monkeypatch.setattr(m, "search_conversations", search)
        result = await _service().list_conversations(db, user_a, 1, 10, keyword="RIDCP")

        assert result.list[0].matched_message_id == 13

    async def test_title_hit_does_not_backfill(self, db, monkeypatch):
        user_a, _ = await _seed_two_conversations(db)

        async def search(user_id, query, *, status, page, size):
            return [1], 1

        monkeypatch.setattr(m, "search_conversations", search)
        result = await _service().list_conversations(db, user_a, 1, 10, keyword="雾霾")

        assert result.list[0].matched_message_id is None

    async def test_no_message_hit_leaves_empty(self, db, monkeypatch):
        _, user_b = await _seed_two_conversations(db)

        async def search(user_id, query, *, status, page, size):
            return [2], 1

        monkeypatch.setattr(m, "search_conversations", search)
        result = await _service().list_conversations(db, user_b, 1, 10, keyword="RIDCP")

        assert result.list[0].matched_message_id is None

    async def test_without_keyword_no_matched_message(self, db):
        user_a, _ = await _seed_two_conversations(db)
        result = await _service().list_conversations(db, user_a, 1, 10)
        assert all(c.matched_message_id is None for c in result.list)

    async def test_deleted_message_not_matched(self, db):
        user_a, _ = await _seed_users(db)
        db.add(_conv(1, user_a, "会话"))
        db.add_all(
            [
                _msg(11, 1, "assistant", "RIDCP 内容", status=2, deleted=1),
                _msg(12, 1, "assistant", "普通回复", status=2),
            ]
        )
        await db.flush()

        matched = await ai_message_repository.find_latest_ids_by_keyword(db, [1], "RIDCP")
        assert matched == {}

    async def test_like_wildcard_not_escaped_as_pattern(self, db):
        """关键词含 % 时按字面量匹配，不放大为全匹配"""
        user_a, _ = await _seed_users(db)
        db.add(_conv(1, user_a, "会话"))
        db.add(_msg(11, 1, "assistant", "普通回复", status=2))
        await db.flush()

        matched = await ai_message_repository.find_latest_ids_by_keyword(db, [1], "%")
        assert matched == {}


class TestAdminConversationDetail:
    async def test_admin_detail_reads_any_user_conversation(self, db):
        _, user_b = await _seed_two_conversations(db)
        result = await _service().get_conversation(db, 2, 0, admin=True)

        assert result.user_id == user_b
        assert result.user_name == "lisi"
        assert result.token_consumed == 0
        assert result.credits_consumed == 0

    async def test_admin_detail_missing_conversation_raises(self, db):
        with pytest.raises(BusinessException) as exc:
            await _service().get_conversation(db, 404, 0, admin=True)
        assert exc.value.code == m.ResultCode.RESOURCE_NOT_FOUND

    async def test_user_detail_keeps_ownership_filter(self, db):
        user_a, _ = await _seed_two_conversations(db)
        with pytest.raises(BusinessException):
            await _service().get_conversation(db, 2, user_a)

    async def test_user_detail_does_not_attach_audit_fields(self, db):
        user_a, _ = await _seed_two_conversations(db)
        result = await _service().get_conversation(db, 1, user_a)

        assert result.user_name is None
        assert result.token_consumed is None
