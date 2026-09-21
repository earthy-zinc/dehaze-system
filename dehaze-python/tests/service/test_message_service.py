"""消息通知服务层测试（真实 MySQL 测试库 + fakeredis）。

覆盖：发送幂等（bizModule+bizId）、模板渲染（变量缺失/禁用/优先级回退）、
未读计数不变量（未读+已读=总数，固定 seed）、越权防护（他人消息不可见/不可操作）、
软删口径、对抗性脏语料持久化、搜索 LIKE 通配符安全、留存期限。

对应消息通知模块测试用例.md 与需求规格 §5.2 安全要求。
"""

from datetime import datetime

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.repository.message_repository import message_repository
from app.service.message_service import message_service

pytestmark = pytest.mark.requires_db

USER_A = 1
USER_B = 2

# 种子模板（config/sql/data/sys_message_template.sql）：business 类型、priority=2
TPL_FEEDBACK_REPLY = "feedback_reply"
# 种子模板：member 类型、priority=3（验证未显式传 priority 时回退模板优先级）
TPL_MEMBER_LEVEL_UP = "member_level_up"


def _send_data(**overrides) -> dict:
    data = {
        "type": "business",
        "title": "svc测试消息",
        "content": "svc测试正文",
        "recipientIds": [USER_A],
    }
    data.update(overrides)
    return data


async def _send(db, **overrides) -> list[int]:
    return await message_service.send(db, _send_data(**overrides))


# ===== 发送与幂等 =====


async def test_send_persists_message(db):
    ids = await _send(db, title="持久化标题", content="持久化正文")
    assert len(ids) == 1
    msg = await message_repository.get_by_id_and_recipient(db, ids[0], USER_A)
    assert msg is not None
    assert msg.title == "持久化标题"
    assert msg.content == "持久化正文"
    assert msg.read_status == 0
    assert msg.deleted == 0
    assert msg.sender_type == 1


async def test_send_expires_at_by_type(db):
    from datetime import timedelta

    alert_ids = await _send(db, type="alert")
    critical_ids = await _send(db, type="critical_alert")
    normal_ids = await _send(db, type="business")
    now = datetime.now()
    # expires_at 列精度为秒，MySQL 落库时对微秒四舍五入（最多 +1s），上界必须留 1s 容差
    rounding = timedelta(seconds=1)

    alert = await message_repository.get_by_id_and_recipient(db, alert_ids[0], USER_A)
    assert alert is not None
    assert alert.expires_at is not None
    assert timedelta(days=6) < alert.expires_at - now <= timedelta(days=7) + rounding

    critical = await message_repository.get_by_id_and_recipient(db, critical_ids[0], USER_A)
    assert critical is not None
    assert critical.expires_at is not None
    assert timedelta(days=89) < critical.expires_at - now <= timedelta(days=90) + rounding

    normal = await message_repository.get_by_id_and_recipient(db, normal_ids[0], USER_A)
    assert normal is not None
    assert normal.expires_at is not None
    assert timedelta(days=29) < normal.expires_at - now <= timedelta(days=30) + rounding


async def test_send_idempotent_by_biz_key_returns_existing_ids(db):
    data = _send_data(bizModule="test", bizId="svc_idem_1")
    first = await message_service.send(db, data)
    second = await message_service.send(db, data)
    assert second == first

    # 不同接收人不去重
    third = await message_service.send(
        db, _send_data(bizModule="test", bizId="svc_idem_1", recipientIds=[USER_B])
    )
    assert len(third) == 1
    assert third[0] not in first


async def test_send_without_biz_key_does_not_dedup(db):
    first = await _send(db)
    second = await _send(db)
    assert first[0] != second[0]


async def test_send_rejects_missing_title_or_content(db):
    with pytest.raises(BusinessException) as ei:
        await _send(db, title=None)
    assert ei.value.code == ResultCode.PARAM_ERROR

    with pytest.raises(BusinessException) as ei:
        await _send(db, content=None)
    assert ei.value.code == ResultCode.PARAM_ERROR


# ===== 模板渲染 =====


async def test_send_with_template_renders_variables(db):
    ids = await message_service.send(
        db,
        _send_data(
            templateCode=TPL_FEEDBACK_REPLY,
            variables={"title": "我的反馈"},
        ),
    )
    msg = await message_repository.get_by_id_and_recipient(db, ids[0], USER_A)
    assert msg is not None
    assert msg.title == "您的反馈「我的反馈」已被管理员回复"
    assert "「我的反馈」" in msg.content


async def test_send_with_template_priority_fallback(db):
    """未显式传 priority 时回退模板优先级（member_level_up priority=3）"""
    ids = await message_service.send(
        db,
        _send_data(
            type="member",
            templateCode=TPL_MEMBER_LEVEL_UP,
            variables={"levelName": "VIP2", "benefitList": "- 权益A"},
        ),
    )
    msg = await message_repository.get_by_id_and_recipient(db, ids[0], USER_A)
    assert msg is not None
    assert msg.priority == 3


async def test_send_with_explicit_priority_overrides_template(db):
    """显式传 priority 时不被模板优先级覆盖"""
    ids = await message_service.send(
        db,
        _send_data(
            type="member",
            priority=1,
            templateCode=TPL_MEMBER_LEVEL_UP,
            variables={"levelName": "VIP2", "benefitList": "- 权益A"},
        ),
    )
    msg = await message_repository.get_by_id_and_recipient(db, ids[0], USER_A)
    assert msg is not None
    assert msg.priority == 1


async def test_send_template_missing_variable_rejected(db):
    with pytest.raises(BusinessException) as ei:
        await message_service.send(db, _send_data(templateCode=TPL_FEEDBACK_REPLY, variables={}))
    assert ei.value.code == ResultCode.TEMPLATE_VAR_MISSING


async def test_send_template_not_found(db):
    with pytest.raises(BusinessException) as ei:
        await message_service.send(
            db, _send_data(templateCode="no_such_template_xyz", variables={})
        )
    assert ei.value.code == ResultCode.MESSAGE_TEMPLATE_NOT_FOUND


async def test_template_variables_hostile_corpus_no_double_expansion(db):
    """变量值含 HTML/emoji/CRLF/嵌套占位符：原样替换、不二次展开（无 {var} 递归渲染）"""
    hostile = "<script>alert(1)</script>\r\n🎉\u200b{title}{title}"
    ids = await message_service.send(
        db,
        _send_data(templateCode=TPL_FEEDBACK_REPLY, variables={"title": hostile}),
    )
    msg = await message_repository.get_by_id_and_recipient(db, ids[0], USER_A)
    assert msg is not None
    # 单遍替换：变量值中的 {title} 保留原样，不再展开
    assert msg.title == f"您的反馈「{hostile}」已被管理员回复"
    assert msg.title.count("{title}") == 2


# ===== 未读计数不变量 =====


async def test_unread_count_invariant_fixed_seed(db):
    """固定 seed：发 3 条 → 未读 3；已读 1 条 → 未读 2；未读+已读=总数恒成立"""
    seed = "unread_invariant_svc"
    ids = []
    for i in range(3):
        ids.extend(await _send(db, bizModule="test", bizId=f"{seed}_{i}"))

    assert await message_service.get_unread_count(db, USER_A) == 3

    await message_service.mark_read(db, USER_A, ids[0])
    unread = await message_service.get_unread_count(db, USER_A)
    assert unread == 2

    await message_service.mark_all_read(db, USER_A, "business")
    assert await message_service.get_unread_count(db, USER_A) == 0

    items, total = await message_repository.get_page(db, USER_A, 1, 100)
    assert total == 3
    assert all(m.read_status == 1 for m in items)


async def test_mark_read_idempotent_and_zero_row_silent(db):
    """重复已读幂等；不存在/他人消息标记静默成功（0 行更新，不暴露存在性）"""
    (mid,) = await _send(db)
    assert await message_service.mark_read(db, USER_A, mid) is True
    assert await message_service.mark_read(db, USER_A, mid) is False
    # 他人消息 / 不存在消息：静默
    assert await message_service.mark_read(db, USER_B, mid) is False
    assert await message_service.mark_read(db, USER_A, 999999999) is False


# ===== 越权防护 =====


async def test_detail_of_other_user_message_not_found(db):
    (mid,) = await _send(db, recipientIds=[USER_B])
    with pytest.raises(BusinessException) as ei:
        await message_service.get_detail(db, USER_A, mid)
    assert ei.value.code == ResultCode.MESSAGE_NOT_FOUND
    # 本人可查
    detail = await message_service.get_detail(db, USER_B, mid)
    assert detail["id"] == mid


async def test_delete_other_user_message_is_noop(db):
    (mid,) = await _send(db, recipientIds=[USER_B])
    # A 尝试删除 B 的消息：无行受影响，B 的消息完好
    await message_service.delete_by_ids(db, USER_A, [mid])
    msg = await message_repository.get_by_id_and_recipient(db, mid, USER_B)
    assert msg is not None
    assert msg.deleted == 0


async def test_delete_marks_soft_deleted_and_excludes_from_list(db):
    (mid,) = await _send(db)
    await message_service.delete_by_ids(db, USER_A, [mid])
    assert await message_repository.get_by_id_and_recipient(db, mid, USER_A) is None
    _, total = await message_repository.get_page(db, USER_A, 1, 100)
    assert total == 0


# ===== 对抗性脏语料持久化 =====


async def test_send_hostile_title_and_content_roundtrip(db):
    dirty_title = "🎉<b>标题</b>\u200b全角ＡＢＣ\r\n" + "长" * 200
    dirty_content = "零宽\u200bBOM\ufeff\tCRLF\r\n${process.env}\n%LIKE_%LIKE\\%" + " emoji🎉" * 50
    (mid,) = await _send(db, title=dirty_title, content=dirty_content)
    msg = await message_repository.get_by_id_and_recipient(db, mid, USER_A)
    assert msg is not None
    assert msg.title == dirty_title
    assert msg.content == dirty_content


async def test_search_like_wildcards_treated_literally(db):
    """%/_ 等通配符按字面匹配，不产生全表模糊命中或异常"""
    marker = "WILD 100% uniquely_9f3a"
    (mid,) = await _send(db, title=marker)
    items, _ = await message_repository.search(db, USER_A, "%_", 1, 100)
    assert all(item.id != mid for item in items)
    items, _ = await message_repository.search(db, USER_A, "uniquely_9f3a", 1, 100)
    assert any(item.id == mid for item in items)


async def test_send_disabled_template_rejected(db):
    """禁用模板：临时将种子模板置 0，断言 A0558 后恢复（测试事务回滚，零污染）"""
    from app.repository.message_template_repository import message_template_repository

    tpl = await message_template_repository.get_by_code(db, TPL_FEEDBACK_REPLY)
    assert tpl is not None
    tpl.status = 0
    with pytest.raises(BusinessException) as ei:
        await message_service.send(
            db, _send_data(templateCode=TPL_FEEDBACK_REPLY, variables={"title": "x"})
        )
    assert ei.value.code == ResultCode.TEMPLATE_DISABLED
