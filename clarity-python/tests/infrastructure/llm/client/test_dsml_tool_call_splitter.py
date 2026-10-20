"""_DsmlToolCallSplitter 文本化工具调用（DSML）兜底解析测试"""

import json

from app.infrastructure.llm.client.openai_compat_client import _DsmlToolCallSplitter

_SAMPLE = (
    "<｜DSML｜tool_calls>\n"
    '<｜DSML｜invoke name="glob">\n'
    '<｜DSML｜parameter name="pattern" string="true">**/*.md</｜DSML｜parameter>\n'
    '<｜DSML｜parameter name="path" string="true">/data/workspace</｜DSML｜parameter>\n'
    "</｜DSML｜invoke>\n"
    "</｜DSML｜tool_calls>"
)


def _run(chunks: list[str]) -> list[tuple[str, str, str]]:
    splitter = _DsmlToolCallSplitter()
    out: list[tuple[str, str, str]] = []
    for c in chunks:
        out.extend(splitter.feed(c))
    out.extend(splitter.flush())
    return out


def _texts(out: list[tuple[str, str, str]]) -> str:
    return "".join(first for kind, first, _ in out if kind == "text")


def _calls(out: list[tuple[str, str, str]]) -> list[tuple[str, dict]]:
    return [(first, json.loads(second)) for kind, first, second in out if kind == "tool_call"]


def _chunk_every(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def test_plain_text_passthrough():
    """无控制符的纯正文原样输出，不产生工具调用"""
    out = _run(["你好", "，世界"])
    assert _texts(out) == "你好，世界"
    assert _calls(out) == []


def test_dsml_parsed_across_chunks():
    """控制符被流式分块拆开时仍解析为一次完整工具调用"""
    out = _run(_chunk_every(_SAMPLE, 7))
    assert _calls(out) == [("glob", {"pattern": "**/*.md", "path": "/data/workspace"})]
    assert _texts(out) == ""


def test_text_around_dsml_preserved():
    """DSML 前后的正文均保留，控制符本身不透出"""
    out = _run(["前缀", _SAMPLE, "后缀"])
    assert _texts(out) == "前缀后缀"
    assert _calls(out) == [("glob", {"pattern": "**/*.md", "path": "/data/workspace"})]


def test_multiple_invokes_in_one_block():
    """一个 tool_calls 块内的多个 invoke 各自解析"""
    sample = (
        "<｜DSML｜tool_calls>"
        '<｜DSML｜invoke name="read_file">'
        '<｜DSML｜parameter name="file_path" string="true">/a.md</｜DSML｜parameter>'
        "</｜DSML｜invoke>"
        '<｜DSML｜invoke name="ls">'
        '<｜DSML｜parameter name="path" string="true">/b</｜DSML｜parameter>'
        "</｜DSML｜invoke>"
        "</｜DSML｜tool_calls>"
    )
    out = _run(_chunk_every(sample, 5))
    assert _calls(out) == [("read_file", {"file_path": "/a.md"}), ("ls", {"path": "/b"})]


def test_unclosed_block_parsed_at_flush():
    """缺少收尾标记（流被截断）时，已完整的 invoke 仍在 flush 时解析"""
    truncated = _SAMPLE.removesuffix("\n</｜DSML｜tool_calls>")
    out = _run(_chunk_every(truncated, 6))
    assert _calls(out) == [("glob", {"pattern": "**/*.md", "path": "/data/workspace"})]


def test_marker_without_invoke_downgraded_to_text():
    """只有控制符、没有可解析 invoke 时按原文降级为正文，不丢内容、不中断会话"""
    out = _run(["<｜DSML｜tool_calls>"])
    assert _calls(out) == []
    assert _texts(out) == "<｜DSML｜tool_calls>"


def test_partial_marker_at_stream_end_flushed_as_text():
    """流结束时尾部残留的控制符前缀按正文输出（避免吞字符）"""
    out = _run(["正文<｜DSML"])
    assert _calls(out) == []
    assert _texts(out) == "正文<｜DSML"
