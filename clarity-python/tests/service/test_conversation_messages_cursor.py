"""会话消息列表游标分页（契约 A）：before 游标 / limit / hasMore / 空会话 / admin / 不重不漏。

self 视角已改为"当前激活链"：分页在 current_branch_message_id 回溯出的链上切，
before 为游标（仅返回 id < before），hasMore 表示链上是否还存在更早消息，total 为链长。
admin 视角保持全量分支分页（total 为会话消息总数）。
"""

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_message import SysAiMessage
from app.repository.ai_message_repository import ai_message_repository
from app.service.ai_conversation_service import ai_conversation_service

pytestmark = pytest.mark.requires_db


async def _seed_conv(db, user_id=1) -> SysAiConversation:
    conv = SysAiConversation(user_id=user_id, model="gpt", agent_code=None)
    db.add(conv)
    await db.flush()
    return conv


async def _add_msg(db, conv_id, parent_id, role, text="", **overrides) -> SysAiMessage:
    fields = {
        "conversation_id": conv_id,
        "parent_message_id": parent_id,
        "role": role,
        "status": "completed",
        "blocks": [{"id": f"b{text}", "kind": "text", "seq": 1, "text": text}],
    }
    fields.update(overrides)
    msg = SysAiMessage(**fields)
    db.add(msg)
    await db.flush()
    return msg


async def _seed_messages(db, conv_id, count, start=None):
    """顺序落库 count 条消息，返回按落库顺序（id 升序）的消息列表"""
    msgs = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append(
            await _add_msg(
                db,
                conv_id,
                msgs[-1].id if msgs else None,
                role,
                text=f"m{i}",
            )
        )
    return msgs


async def _set_branch(db, conv, msg: SysAiMessage) -> None:
    """推进分支指针（与 send_message/switch_branch 落库后的会话状态一致）"""
    conv.current_branch_message_id = msg.id
    await db.flush()


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

        # 并发插入更高 id 的新消息：真实发送会挂在激活链尾并推进分支指针
        new_msg = await _add_msg(db, conv.id, msgs[-1].id, "user", text="new")
        await _set_branch(db, conv, new_msg)

        page2 = await _list(db, conv.id, page1.list[-1].id, 2)
        assert [m.id for m in page2.list] == [msgs[2].id, msgs[1].id]
        assert page2.hasMore is True

        page3 = await _list(db, conv.id, page2.list[-1].id, 2)
        assert [m.id for m in page3.list] == [msgs[0].id]
        assert page3.hasMore is False

        collected = [m.id for m in page1.list + page2.list + page3.list]
        assert collected == [m.id for m in reversed(msgs)]  # 恰好每条一次（新消息不在旧游标范围内）


class TestMessageShapeAndAdmin:
    async def test_messages_use_contract_shape(self, db):
        """列表出参是契约 Message：内容在 blocks、计划/中断挂在消息上、无旧字段"""
        conv = await _seed_conv(db)
        msgs = await _seed_messages(db, conv.id, 2)
        msgs[1].blocks = [
            {"id": "t1", "kind": "thinking", "seq": 1, "text": "先想", "durationMs": 3},
            {"id": "b1", "kind": "text", "seq": 1, "text": "答案", "durationMs": 5},
        ]
        msgs[1].status = "awaiting_input"
        msgs[1].pending_interrupt = {
            "id": "i1",
            "type": "confirm",
            "createdAt": "2026-01-01T00:00:00",
            "expiresAt": "2099-01-01T00:00:00",
            "actionable": True,
        }
        await db.flush()

        result = await _list(db, conv.id, None, 10)
        assistant = next(m for m in result.list if m.id == msgs[1].id)
        assert [b.kind for b in assistant.blocks] == ["thinking", "text"]
        assert assistant.blocks[1].text == "答案"
        assert assistant.pendingInterrupt is not None
        assert assistant.pendingInterrupt.type == "confirm"
        assert assistant.status == "awaiting_input"
        # 旧投影不再出现在出参
        dumped = assistant.model_dump(by_alias=True)
        assert "content" not in dumped
        assert "thoughts" not in dumped

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


class TestActiveChain:
    """self 视角返回激活链：指针可信，指针失效时回退，不因分支/软删返回空列表。"""

    async def test_returns_only_current_branch(self, db):
        conv = await _seed_conv(db)
        u0 = await _add_msg(db, conv.id, None, "user", text="u0")
        a1 = await _add_msg(db, conv.id, u0.id, "assistant", text="a1")
        b1 = await _add_msg(db, conv.id, u0.id, "assistant", text="b1")
        u2 = await _add_msg(db, conv.id, a1.id, "user", text="u2")
        await _set_branch(db, conv, u2)

        result = await _list(db, conv.id, None, 50)

        assert [m.id for m in result.list] == [u2.id, a1.id, u0.id]
        assert result.total == 3
        assert result.hasMore is False

        # admin 仍看全量分支，审计不受 self 链裁剪影响
        admin_result = await _list(db, conv.id, None, 50, admin=True)
        assert admin_result.total == 4
        assert b1.id in [m.id for m in admin_result.list]
        assert all(m.siblingIndex is None and m.siblingCount is None for m in admin_result.list)

    async def test_null_pointer_falls_back_to_latest_message(self, db):
        conv = await _seed_conv(db)
        u0 = await _add_msg(db, conv.id, None, "user", text="u0")
        await _add_msg(db, conv.id, u0.id, "assistant", text="a1")
        b1 = await _add_msg(db, conv.id, u0.id, "assistant", text="b1")
        # 指针为空（历史会话/初始化未落指针）：回退到最大 id 消息所在的链
        result = await _list(db, conv.id, None, 50)

        assert [m.id for m in result.list] == [b1.id, u0.id]
        assert result.total == 2

    async def test_pointer_to_deleted_message_falls_back(self, db):
        conv = await _seed_conv(db)
        u0 = await _add_msg(db, conv.id, None, "user", text="u0")
        a1 = await _add_msg(db, conv.id, u0.id, "assistant", text="a1")
        a2 = await _add_msg(db, conv.id, u0.id, "assistant", text="a2")
        await _set_branch(db, conv, a1)
        await ai_message_repository.soft_delete_by_ids(db, [a1.id])

        result = await _list(db, conv.id, None, 50)

        # 回退到最大 id 的未删消息（a2），而不是返回空列表
        assert [m.id for m in result.list] == [a2.id, u0.id]
        assert result.total == 2

    async def test_dangling_pointer_falls_back(self, db):
        conv = await _seed_conv(db)
        u0 = await _add_msg(db, conv.id, None, "user", text="u0")
        a1 = await _add_msg(db, conv.id, u0.id, "assistant", text="a1")
        conv.current_branch_message_id = 9_999_999
        await db.flush()

        result = await _list(db, conv.id, None, 50)

        assert [m.id for m in result.list] == [a1.id, u0.id]
        assert result.total == 2

    async def test_deleted_middle_node_breaks_chain(self, db):
        conv = await _seed_conv(db)
        u0 = await _add_msg(db, conv.id, None, "user", text="u0")
        a1 = await _add_msg(db, conv.id, u0.id, "assistant", text="a1")
        u2 = await _add_msg(db, conv.id, a1.id, "user", text="u2")
        a3 = await _add_msg(db, conv.id, u2.id, "assistant", text="a3")
        await _set_branch(db, conv, a3)
        await ai_message_repository.soft_delete_by_ids(db, [a1.id])

        result = await _list(db, conv.id, None, 50)

        assert [m.id for m in result.list] == [a3.id, u2.id]
        assert result.total == 2

    async def test_sibling_index_and_count(self, db):
        conv = await _seed_conv(db)
        u0 = await _add_msg(db, conv.id, None, "user", text="u0")
        a1 = await _add_msg(db, conv.id, u0.id, "assistant", text="a1")
        a2 = await _add_msg(db, conv.id, u0.id, "assistant", text="a2")
        u3 = await _add_msg(db, conv.id, a2.id, "user", text="u3")
        a4 = await _add_msg(db, conv.id, u3.id, "assistant", text="a4")
        await _set_branch(db, conv, a4)

        result = await _list(db, conv.id, None, 50)
        by_id = {m.id: m for m in result.list}

        assert (by_id[a2.id].siblingIndex, by_id[a2.id].siblingCount) == (2, 2)
        assert (by_id[a4.id].siblingIndex, by_id[a4.id].siblingCount) == (1, 1)
        # 非 assistant / 不在激活链的兄弟不返回
        assert by_id[u0.id].siblingIndex is None
        assert by_id[u3.id].siblingIndex is None
        assert a1.id not in by_id

        # 软删兄弟后序号与总数同步收缩
        await ai_message_repository.soft_delete_by_ids(db, [a1.id])
        result = await _list(db, conv.id, None, 50)
        a2_view = next(m for m in result.list if m.id == a2.id)
        assert (a2_view.siblingIndex, a2_view.siblingCount) == (1, 1)

    async def test_root_assistant_has_null_siblings(self, db):
        conv = await _seed_conv(db)
        a0 = await _add_msg(db, conv.id, None, "assistant", text="root")
        await _set_branch(db, conv, a0)

        result = await _list(db, conv.id, None, 50)

        assert [m.id for m in result.list] == [a0.id]
        assert result.list[0].siblingIndex is None
        assert result.list[0].siblingCount is None

    async def test_chain_pagination_walks_backwards(self, db):
        conv = await _seed_conv(db)
        msgs = await _seed_messages(db, conv.id, 6)
        await _set_branch(db, conv, msgs[-1])

        page1 = await _list(db, conv.id, None, 2)
        assert [m.id for m in page1.list] == [msgs[5].id, msgs[4].id]
        assert page1.total == 6
        assert page1.hasMore is True

        page2 = await _list(db, conv.id, page1.list[-1].id, 2)
        assert [m.id for m in page2.list] == [msgs[3].id, msgs[2].id]
        assert page2.total == 6
        assert page2.hasMore is True

        page3 = await _list(db, conv.id, page2.list[-1].id, 2)
        assert [m.id for m in page3.list] == [msgs[1].id, msgs[0].id]
        assert page3.hasMore is False

        collected = [m.id for m in page1.list + page2.list + page3.list]
        assert collected == [m.id for m in reversed(msgs)]
