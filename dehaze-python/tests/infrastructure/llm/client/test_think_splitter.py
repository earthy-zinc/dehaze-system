"""_ThinkSplitter 跨块思考标签切分测试"""

from app.infrastructure.llm.client.openai_compat_client import _ThinkSplitter


def _run(chunks: list[str]) -> list[tuple[str, str]]:
    splitter = _ThinkSplitter()
    out: list[tuple[str, str]] = []
    for c in chunks:
        out.extend(splitter.feed(c))
    out.extend(splitter.flush())
    return out


def _agg(chunks: list[str]) -> list[tuple[str, str]]:
    """按相邻同类型段聚合（流式逐块分段 yield 属正常语义，断言聚合结果）"""
    out: list[tuple[str, str]] = []
    for kind, seg in _run(chunks):
        if out and out[-1][0] == kind:
            out[-1] = (kind, out[-1][1] + seg)
        else:
            out.append((kind, seg))
    return out


def test_think_block_split_across_chunks():
    """标签被流式分块拆开时仍正确切分：思考段与正文段各自完整"""
    assert _agg(["<thi", "nk>用户想要", "一个答", "案</thi", "nk>答案是42"]) == [
        ("thinking", "用户想要一个答案"),
        ("text", "答案是42"),
    ]


def test_plain_text_passthrough():
    """无标签的纯正文按序完整输出"""
    assert _agg(["你好", "，", "世界"]) == [("text", "你好，世界")]


def test_literal_angle_bracket_kept():
    """正文含字面量 '<'（非标签前缀）时不丢内容"""
    assert _agg(["a < b", " 且 3<5"]) == [("text", "a < b 且 3<5")]


def test_partial_prefix_at_stream_end_flushed():
    """流结束时尾部残留的不完整标签前缀按正文输出（模型字面量输出 <thi 场景）"""
    assert _agg(["正文<thi"]) == [("text", "正文<thi")]


def test_unclosed_think_treated_as_text():
    """think 未闭合（小模型对 /no_think 不遵从的常见形态）时内容降级为正文，避免空回复"""
    assert _agg(["<think>只输出了一半"]) == [("text", "只输出了一半")]


def test_multiple_think_blocks():
    """多个 think 块交替出现均正确切分"""
    assert _agg(["<think>一</think>甲<think>二</think>乙"]) == [
        ("thinking", "一"),
        ("text", "甲"),
        ("thinking", "二"),
        ("text", "乙"),
    ]


def test_think_open_split_exactly():
    """标签恰好整块到达的边界场景"""
    assert _agg(["<think>", "思考", "</think>", "正文"]) == [
        ("thinking", "思考"),
        ("text", "正文"),
    ]


def test_think_content_delivered_on_close():
    """思考内容缓冲至闭合标签到达时一次性下发，不逐块提前输出"""
    assert _agg(["<think>思", "考</think>正文"]) == [
        ("thinking", "思考"),
        ("text", "正文"),
    ]
