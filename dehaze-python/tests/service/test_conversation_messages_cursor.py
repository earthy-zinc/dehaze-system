"""会话消息列表游标分页（契约 A）：before 游标 / limit / hasMore / 空会话 / admin / 不重不漏。

替换历史 pageNum/pageSize 分页：按 id 倒序取一页（id 单调等价时间倒序），
before 为游标（仅返回 id < before），hasMore 表示是否还存在更早消息。
"""

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_agent_thought import SysAiAgentThought
from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_message import SysAiMessage
from app.service.ai_conversation_service import ai_conversation_service

pytestmark = pytest.mark.requires_db


async def _seed_conv(db, user_id=1) -> SysAiConversation:
    conv = SysAiConversation(user_id=user_id, model="gpt", agent_code=None)
    db.add(conv)
    await db.flush()
    return conv


async def _seed_messages(db, conv_id, count, start=None):
    """顺序落库 count 条消息，返回按落库顺序（id 升序）的消息列表"""
    msgs = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        m = SysAiMessage(
            conversation_id=conv_id,
            parent_message_id=msgs[-1].id if msgs else None,
            role=role,
            content=f"m{i}",
            status=2,
        )
        db.add(m)
        await db.flush()
        msgs.append(m)
    return msgs


async def _list(db, conv_id, before, limit, admin=False, user_id=1):
    return await ai_conversation_service.list_messages(
        db, conv_id, user_id, before, limit, admin=admin
    )


class TestCursorPaging:
    async def test_default_before_takes_latest_page(self, db):
        conv = await _seed_conv(db)
        msgs = await _seed_messages(db, conv.id, 5)

        result = await _list(db, conv.id, None, 3)

        assert [m.id for m in result.list] == [msgs[4].id, msgs[3].id, msgs[2].id]
        assert result.total == 5
        assert result.hasMore is True

    async def test_before_cursor_returns_older_page(self, db):
        conv = await _seed_conv(db)
        msgs = await _seed_messages(db, conv.id, 5)

        result = await _list(db, conv.id, msgs[2].id, 3)

        assert [m.id for m in result.list] == [msgs[1].id, msgs[0].id]
        assert result.total == 5
        assert result.hasMore is False

    async def test_before_equal_min_id_returns_empty(self, db):
        conv = await _seed_conv(db)
        msgs = await _seed_messages(db, conv.id, 3)

        result = await _list(db, conv.id, msgs[0].id, 10)

        assert result.list == []
        assert result.total == 3
        assert result.hasMore is False

    async def test_before_nonexistent_id_uses_cursor_boundary(self, db):
        conv = await _seed_conv(db)
        msgs = await _seed_messages(db, conv.id, 5)

        result = await _list(db, conv.id, msgs[-1].id + 100, 2)

        assert [m.id for m in result.list] == [msgs[4].id, msgs[3].id]
        assert result.total == 5
        assert result.hasMore is True

    async def test_has_more_false_when_page_exactly_covers(self, db):
        conv = await _seed_conv(db)
        await _seed_messages(db, conv.id, 5)

        result = await _list(db, conv.id, None, 5)

        assert len(result.list) == 5
        assert result.total == 5
        assert result.hasMore is False

    async def test_limit_one(self, db):
        conv = await _seed_conv(db)
        msgs = await _seed_messages(db, conv.id, 3)

        result = await _list(db, conv.id, None, 1)

        assert [m.id for m in result.list] == [msgs[2].id]
        assert result.hasMore is True

    async def test_empty_conversation(self, db):
        conv = await _seed_conv(db)

        result = await _list(db, conv.id, None, 50)

        assert result.list == []
        assert result.total == 0
        assert result.hasMore is False


class TestCursorConcurrency:
    async def test_no_duplicate_no_gap_with_concurrent_insert(self, db):
        """翻页途中插入新消息：按 before 继续翻页不重复、不丢旧消息"""
        conv = await _seed_conv(db)
        msgs = await _seed_messages(db, conv.id, 5)

        page1 = await _list(db, conv.id, None, 2)
        assert [m.id for m in page1.list] == [msgs[4].id, msgs[3].id]
        assert page1.hasMore is True

        # 并发插入更高 id 的新消息（不应影响历史页）
        new_msg = SysAiMessage(
            conversation_id=conv.id, parent_message_id=None, role="user", content="new", status=2
        )
        db.add(new_msg)
        await db.flush()

        page2 = await _list(db, conv.id, page1.list[-1].id, 2)
        assert [m.id for m in page2.list] == [msgs[2].id, msgs[1].id]
        assert page2.hasMore is True

        page3 = await _list(db, conv.id, page2.list[-1].id, 2)
        assert [m.id for m in page3.list] == [msgs[0].id]
        assert page3.hasMore is False

        collected = [m.id for m in page1.list + page2.list + page3.list]
        assert collected == [m.id for m in reversed(msgs)]  # 恰好每条一次（新消息不在旧游标范围内）


class TestThoughtsAndAdmin:
    async def test_assistant_messages_carry_thoughts(self, db):
        """思考链组装逻辑不变：assistant 消息批量附带推理步骤（position 正序）"""
        conv = await _seed_conv(db)
        msgs = await _seed_messages(db, conv.id, 2)  # m0=user, m1=assistant
        for pos, text in ((2, "第二步"), (1, "第一步")):
            db.add(
                SysAiAgentThought(
                    message_id=msgs[1].id,
                    conversation_id=conv.id,
                    position=pos,
                    thought=text,
                    status=1,
                    latency_ms=0,
                )
            )
        await db.flush()

        result = await _list(db, conv.id, None, 10)
        assistant = next(m for m in result.list if m.id == msgs[1].id)
        assert assistant.thoughts is not None
        assert [t.thought for t in assistant.thoughts] == ["第一步", "第二步"]

    async def test_admin_reads_arbitrary_user_conversation(self, db):
        conv = await _seed_conv(db, user_id=99)
        await _seed_messages(db, conv.id, 2)

        admin_result = await _list(db, conv.id, None, 10, admin=True, user_id=1)
        assert admin_result.total == 2

        with pytest.raises(BusinessException) as ei:
            await _list(db, conv.id, None, 10, admin=False, user_id=1)
        assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND

    async def test_missing_conversation_raises(self, db):
        with pytest.raises(BusinessException) as ei:
            await _list(db, 999999, None, 10)
        assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND
