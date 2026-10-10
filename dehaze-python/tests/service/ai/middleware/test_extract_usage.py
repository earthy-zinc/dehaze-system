"""_extract_usage 多轮 usage 汇总测试（计费结算口径）"""

from langchain_core.messages import AIMessage, HumanMessage

from app.service.ai.middleware.dehaze_hooks_middleware import _extract_usage


def _ai(usage: dict | None, content: str = "x") -> AIMessage:
    return AIMessage(content=content, response_metadata={"usage": usage} if usage else {})


def test_empty_messages():
    assert _extract_usage(None) == {}
    assert _extract_usage([]) == {}


def test_single_call_usage_passthrough():
    messages = [HumanMessage(content="hi"), _ai({"input_tokens": 10, "output_tokens": 5})]
    assert _extract_usage(messages) == {"input_tokens": 10, "output_tokens": 5}


def test_multi_round_usage_summed():
    """多轮工具循环的各次调用 usage 必须累加，而非只取最后一条"""
    messages = [
        HumanMessage(content="hi"),
        _ai({"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 30}),
        _ai({"input_tokens": 200, "output_tokens": 40, "cached_input_tokens": 10}),
    ]
    assert _extract_usage(messages) == {
        "input_tokens": 300,
        "output_tokens": 60,
        "cached_input_tokens": 40,
    }


def test_ai_without_usage_ignored():
    """无 usage 的消息（如提示词拦截返回）不参与汇总，也不清零已累计值"""
    messages = [_ai({"input_tokens": 7, "output_tokens": 3}), _ai(None)]
    assert _extract_usage(messages) == {"input_tokens": 7, "output_tokens": 3}


def test_all_without_usage_returns_empty():
    assert _extract_usage([HumanMessage(content="hi"), _ai(None)]) == {}


def test_non_numeric_usage_value_kept_once():
    """非数值字段（如路由归因字符串）取首个，不参与求和"""
    messages = [
        _ai({"input_tokens": 1, "route": "primary"}),
        _ai({"input_tokens": 2, "route": "fallback"}),
    ]
    assert _extract_usage(messages) == {"input_tokens": 3, "route": "primary"}
