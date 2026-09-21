"""ai_conversation_repository 标题关键词 LIKE 转义的真实 SQL 语义测试（MySQL dehaze_test）"""

import pytest

from app.models.entity.sys_ai_conversation import SysAiConversation
from app.repository.ai_conversation_repository import ai_conversation_repository

pytestmark = pytest.mark.requires_db


async def _make_conv(db, title):
    conv = SysAiConversation(user_id=1, title=title, model="m1", status=1)
    db.add(conv)
    await db.flush()
    return conv


async def _titles(db, keyword):
    convs, _total = await ai_conversation_repository.paginate_all_with_keyword(db, 1, 10, keyword)
    return {c.title for c in convs}


async def test_keyword_percent_matches_literal_only(db):
    """% 未转义时会退化成"匹配任意串"，命中全部会话"""
    await _make_conv(db, "普通会话")
    await _make_conv(db, "进度100%完成")

    assert await _titles(db, "%") == {"进度100%完成"}


async def test_keyword_underscore_matches_literal_only(db):
    """_ 未转义时会退化成"匹配任意单字符"，命中含任意字符的标题"""
    await _make_conv(db, "普通会话")
    await _make_conv(db, "模型_a_b")

    assert await _titles(db, "_") == {"模型_a_b"}


async def test_keyword_backslash_matches_literal_only(db):
    await _make_conv(db, "普通会话")
    await _make_conv(db, r"路径C:\tmp")

    assert await _titles(db, "\\") == {r"路径C:\tmp"}


async def test_keyword_plain_substring_matches(db):
    await _make_conv(db, "雾霾成因分析")
    await _make_conv(db, "无关会话")

    assert await _titles(db, "雾霾") == {"雾霾成因分析"}
