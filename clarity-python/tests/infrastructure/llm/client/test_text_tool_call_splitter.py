"""_TextToolCallSplitter 文本化工具调用（text_json 方言）解析测试"""

import json

from app.infrastructure.llm.client.openai_compat_client import _TextToolCallSplitter

_SAMPLE = '<tool_call>{"name": "knowledge_base_search", "arguments": {"query": "分页"}}</tool_call>'


def _run(chunks: list[str]) -> list[tuple[str, str, str]]:
    splitter = _TextToolCallSplitter()
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
    out = _run(["你好", "，世界"])
    assert _texts(out) == "你好，世界"
    assert _calls(out) == []


def test_tool_call_parsed_across_chunks():
    out = _run(_chunk_every(_SAMPLE, 7))
    assert _calls(out) == [("knowledge_base_search", {"query": "分页"})]
    assert _texts(out) == ""


def test_text_around_call_preserved():
    out = _run(["前缀", _SAMPLE, "后缀"])
    assert _texts(out) == "前缀后缀"
    assert _calls(out) == [("knowledge_base_search", {"query": "分页"})]


def test_multiple_calls_in_stream():
    second = '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>'
    out = _run([_SAMPLE, second])
    assert _calls(out) == [
        ("knowledge_base_search", {"query": "分页"}),
        ("web_search", {"query": "x"}),
    ]


def test_invalid_json_downgraded_to_text():
    """非法 JSON 不得中断会话：按原文降级为正文"""
    bad = "<tool_call>{not json}</tool_call>"
    out = _run([bad])
    assert _calls(out) == []
    assert _texts(out) == bad


def test_missing_name_downgraded_to_text():
    bad = '<tool_call>{"arguments": {"a": 1}}</tool_call>'
    out = _run([bad])
    assert _calls(out) == []
    assert _texts(out) == bad


def test_unclosed_call_flushed_as_text():
    out = _run(['<tool_call>{"name": "web_search"'])
    assert _calls(out) == []
    assert _texts(out) == '<tool_call>{"name": "web_search"'


def test_arguments_defaults_to_empty_object():
    out = _run(['<tool_call>{"name": "get_task_status"}</tool_call>'])
    assert _calls(out) == [("get_task_status", {})]


def test_partial_open_marker_at_stream_end_flushed_as_text():
    out = _run(["正文<tool_cal"])
    assert _calls(out) == []
    assert _texts(out) == "正文<tool_cal"
