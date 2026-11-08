"""ai_message_repository：新列（blocks/plan/pending_interrupt/suggestions）读写与分页。

消息只有一种形状：内容在 blocks，计划/中断/推荐问题挂在消息上；状态是字符串枚举。
"""

import pytest

from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_message import SysAiMessage, block_seq_base, blocks_text, text_block
from app.repository.ai_message_repository import ai_message_repository

pytestmark = pytest.mark.requires_db


async def _make_conv(db) -> SysAiConversation:
    conv = SysAiConversation(user_id=1, title="消息仓储测试", model="m1", status=1)
    db.add(conv)
    await db.flush()
    return conv


async def _make_msg(db, conv_id, **overrides) -> SysAiMessage:
    fields = {
        "conversation_id": conv_id,
        "role": "assistant",
        "status": "running",
        "blocks": [],
    }
    fields.update(overrides)
    msg = SysAiMessage(**fields)
    db.add(msg)
    await db.flush()
    return msg


async def test_blocks_roundtrip_and_text_derivation(db):
    conv = await _make_conv(db)
    msg = await _make_msg(
        db,
        conv.id,
        blocks=[
            {"id": "t1", "kind": "thinking", "seq": 1, "text": "先想", "durationMs": 3},
            {"id": "b1", "kind": "text", "seq": 1, "text": "第一段", "durationMs": 5},
            {"id": "b2", "kind": "text", "seq": 2, "text": "第二段", "durationMs": 7},
        ],
    )
    await db.refresh(msg)

    assert [b["id"] for b in msg.blocks] == ["t1", "b1", "b2"]
    assert blocks_text(msg.blocks) == "第一段\n\n第二段"
    assert block_seq_base(msg.blocks) == {"thinking": 1, "text": 2}


async def test_update_blocks_plan_interrupt_suggestions(db):
    conv = await _make_conv(db)
    msg = await _make_msg(db, conv.id)

    await ai_message_repository.update_blocks(db, msg.id, [text_block("回答")])
    await ai_message_repository.update_plan(
        db,
        msg.id,
        {
            "phase": "plan",
            "status": "pending",
            "tasks": [],
            "revisions": [],
            "awaitingApproval": True,
        },
    )
    pending = {
        "id": "i1",
        "type": "confirm",
        "createdAt": "2026-01-01T00:00:00",
        "expiresAt": "2026-01-31T00:00:00",
        "actionable": True,
    }
    await ai_message_repository.update_pending_interrupt(db, msg.id, pending)
    await ai_message_repository.update_suggestions(db, msg.id, ["追问一"])
    await ai_message_repository.update_status(db, msg.id, "awaiting_input")
    await db.flush()
    await db.refresh(msg)

    assert msg.status == "awaiting_input"
    assert msg.text == "回答"
    assert msg.plan["awaitingApproval"] is True
    assert msg.pending_interrupt["id"] == "i1"
    assert msg.suggestions == ["追问一"]

    # 恢复成功后清空中断
    await ai_message_repository.update_pending_interrupt(db, msg.id, None)
    await db.refresh(msg)
    assert msg.pending_interrupt is None


async def test_update_usage_and_memory_ids(db):
    conv = await _make_conv(db)
    msg = await _make_msg(db, conv.id)

    await ai_message_repository.update_usage(
        db, msg.id, input_tokens=11, output_tokens=22, cached_input_tokens=3, credits=9
    )
    await ai_message_repository.update_memory_ids(db, msg.id, [4, 5])
    await db.flush()
    await db.refresh(msg)

    assert (msg.input_tokens, msg.output_tokens, msg.cached_input_tokens) == (11, 22, 3)
    assert msg.credits == 9
    assert msg.used_memory_ids == [4, 5]


async def test_cursor_pagination_and_total(db):
    conv = await _make_conv(db)
    ids = []
    for i in range(5):
        msg = await _make_msg(
            db, conv.id, role="user", status="completed", blocks=[text_block(f"m{i}")]
        )
        ids.append(msg.id)

    rows, total, has_more = await ai_message_repository.list_by_conversation_cursor(
        db, conv.id, before=None, limit=2
    )
    assert [r.id for r in rows] == [ids[4], ids[3]]
    assert total == 5
    assert has_more is True

    rows2, _, has_more2 = await ai_message_repository.list_by_conversation_cursor(
        db, conv.id, before=ids[3], limit=2
    )
    assert [r.id for r in rows2] == [ids[2], ids[1]]
    assert has_more2 is True

    rows3, _, has_more3 = await ai_message_repository.list_by_conversation_cursor(
        db, conv.id, before=ids[1], limit=2
    )
    assert [r.id for r in rows3] == [ids[0]]
    assert has_more3 is False


async def test_anomaly_status_is_string_enum(db):
    conv = await _make_conv(db)
    await _make_msg(db, conv.id, status="failed")
    await _make_msg(db, conv.id, status="canceled")
    await _make_msg(db, conv.id, status="completed")

    statuses = await ai_message_repository.list_anomaly_status_by_conversations(db, [conv.id])
    assert statuses[conv.id] == {"failed", "canceled"}


async def test_keyword_search_hits_blocks_text(db):
    conv = await _make_conv(db)
    await _make_msg(
        db, conv.id, role="user", status="completed", blocks=[text_block("如何去除雾霾")]
    )
    await _make_msg(db, conv.id, role="user", status="completed", blocks=[text_block("无关内容")])

    matched = await ai_message_repository.find_latest_ids_by_keyword(db, [conv.id], "雾霾")
    assert conv.id in matched
    assert await ai_message_repository.find_latest_ids_by_keyword(db, [conv.id], "不存在的词") == {}


async def test_get_max_message_id_ignores_deleted(db):
    conv = await _make_conv(db)
    m1 = await _make_msg(db, conv.id)
    m2 = await _make_msg(db, conv.id)

    assert await ai_message_repository.get_max_message_id(db, conv.id) == m2.id
    await ai_message_repository.soft_delete_by_ids(db, [m2.id])
    assert await ai_message_repository.get_max_message_id(db, conv.id) == m1.id


async def test_assistant_sibling_ids_grouped_excluding_deleted(db):
    """兄弟元数据一次分组：只含未删 assistant、id 升序、不跨会话。"""
    conv = await _make_conv(db)
    user = await _make_msg(db, conv.id, role="user")
    a1 = await _make_msg(db, conv.id, parent_message_id=user.id, role="assistant")
    a2 = await _make_msg(db, conv.id, parent_message_id=user.id, role="assistant")
    a3 = await _make_msg(db, conv.id, parent_message_id=user.id, role="assistant")
    other_conv = await _make_conv(db)
    await _make_msg(db, other_conv.id, parent_message_id=user.id, role="assistant")
    await ai_message_repository.soft_delete_by_ids(db, [a3.id])

    grouped = await ai_message_repository.list_assistant_sibling_ids(db, conv.id, [user.id])
    assert grouped == {user.id: [a1.id, a2.id]}
    assert await ai_message_repository.list_assistant_sibling_ids(db, conv.id, []) == {}


async def test_get_children_filters_by_role(db):
    """分支列表只取 assistant 子消息；删除校验不传 role 时须看到全部子消息。"""
    conv = await _make_conv(db)
    parent = await _make_msg(db, conv.id, role="user")
    a1 = await _make_msg(db, conv.id, parent_message_id=parent.id, role="assistant")
    a2 = await _make_msg(db, conv.id, parent_message_id=parent.id, role="assistant")
    follow_up = await _make_msg(db, conv.id, parent_message_id=parent.id, role="user")
    deleted = await _make_msg(db, conv.id, parent_message_id=parent.id, role="assistant")
    await ai_message_repository.soft_delete_by_ids(db, [deleted.id])

    all_children = await ai_message_repository.get_children(db, conv.id, parent.id)
    assert {m.id for m in all_children} == {a1.id, a2.id, follow_up.id}

    assistants = await ai_message_repository.get_children(db, conv.id, parent.id, role="assistant")
    assert {m.id for m in assistants} == {a1.id, a2.id}


async def test_chain_skips_soft_deleted_nodes(db):
    """链回溯显式过滤软删行：中间节点被删后链在该处断开，链尾自身被删返回空。"""
    conv = await _make_conv(db)
    m0 = await _make_msg(db, conv.id, role="user")
    m1 = await _make_msg(db, conv.id, parent_message_id=m0.id, role="assistant")
    m2 = await _make_msg(db, conv.id, parent_message_id=m1.id, role="user")
    m3 = await _make_msg(db, conv.id, parent_message_id=m2.id, role="assistant")
    await ai_message_repository.soft_delete_by_ids(db, [m1.id])

    chain = await ai_message_repository.get_chain_by_id(db, conv.id, m3.id)
    assert [m.id for m in chain] == [m2.id, m3.id]
    assert await ai_message_repository.get_chain_by_id(db, conv.id, m1.id) == []
