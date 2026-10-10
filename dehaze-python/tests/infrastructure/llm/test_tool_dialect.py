"""工具调用方言解析与提示词渲染测试"""

from app.infrastructure.llm.tool_dialect import (
    DSML,
    NATIVE_ANTHROPIC,
    NATIVE_OPENAI,
    NONE,
    TEXT_JSON,
    is_prompt_tool_family,
    render_tool_prompt,
    resolve_tool_dialect,
)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "knowledge_base_search",
            "description": "检索平台知识库",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
]


class TestResolveToolDialect:
    def test_anthropic_protocol_is_native(self):
        assert resolve_tool_dialect("anthropic", "claude-3", "chat", 1) == NATIVE_ANTHROPIC

    def test_native_openai_when_tool_capable(self):
        assert resolve_tool_dialect("openai_compat", "gpt-4o", "chat", 1) == NATIVE_OPENAI

    def test_deepseek_v4_is_dsml_regardless_of_cap(self):
        assert (
            resolve_tool_dialect("openai_compat", "deepseek-ai/DeepSeek-V4-Flash", "chat", 1)
            == DSML
        )
        assert resolve_tool_dialect("openai_compat", "deepseek-v4-flash", "chat", 0) == DSML

    def test_non_tool_family_gets_text_json(self):
        assert resolve_tool_dialect("openai_compat", "qwen3-0.6b", "chat", 0) == TEXT_JSON

    def test_unknown_non_tool_family_is_none(self):
        assert resolve_tool_dialect("openai_compat", "some-random-model", "chat", 0) == NONE

    def test_non_chat_model_type_is_none(self):
        assert resolve_tool_dialect("openai_compat", "bge-m3", "embedding", 0) == NONE

    def test_prompt_tool_family_helper(self):
        assert is_prompt_tool_family("qwen3-0.6b")
        assert not is_prompt_tool_family("some-random-model")
        assert not is_prompt_tool_family("qwen3-0.6b", "embedding")


class TestRenderToolPrompt:
    def test_text_dialect_renders_protocol_and_tools(self):
        prompt = render_tool_prompt(_TOOLS, TEXT_JSON)
        assert "<tool_call>" in prompt
        assert "knowledge_base_search：检索平台知识库" in prompt
        assert '"query"' in prompt

    def test_native_dialects_render_nothing(self):
        assert render_tool_prompt(_TOOLS, NATIVE_OPENAI) == ""
        assert render_tool_prompt(_TOOLS, NATIVE_ANTHROPIC) == ""
        assert render_tool_prompt(_TOOLS, DSML) == ""

    def test_empty_tools_render_nothing(self):
        assert render_tool_prompt([], TEXT_JSON) == ""
        assert render_tool_prompt(None, TEXT_JSON) == ""
