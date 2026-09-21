package com.pei.dehaze.service.impl;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * 消息搜索 LIKE 转义单测：与 Python escape_like、Go EscapeLike 口径一致（\ % _ 按字面匹配）。
 */
class MessageServiceImplEscapeLikeTest {

    @Test
    void plainKeywordUnchanged() {
        assertEquals("plain", MessageServiceImpl.escapeLike("plain"));
    }

    @Test
    void percentAndUnderscoreEscaped() {
        assertEquals("100\\%", MessageServiceImpl.escapeLike("100%"));
        assertEquals("under\\_score", MessageServiceImpl.escapeLike("under_score"));
    }

    @Test
    void backslashDoubledFirst() {
        assertEquals("back\\\\slash", MessageServiceImpl.escapeLike("back\\slash"));
        // 含字面反斜杠时通配符转义仍正确（双写在前，与 Python 顺序替换结果一致）
        assertEquals("\\%\\\\\\_", MessageServiceImpl.escapeLike("%\\_"));
    }

    @Test
    void nullAndEmptyPassThrough() {
        assertNull(MessageServiceImpl.escapeLike(null));
        assertEquals("", MessageServiceImpl.escapeLike(""));
    }
}
