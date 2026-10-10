"""工具调用方言：决定"工具如何交给模型、调用如何解析回来"。

原生 function calling（OpenAI `tools` 字段 / Anthropic `tools`）由协议客户端直接处理。
当模型本身不支持 function calling，或服务方未实现工具协议解析（如 vLLM/SGLang 未配
`--tool-call-parser`，模型把工具调用当正文文本输出，见 DeepSeek V4 的 DSML）时，退化为
"把工具清单按文本协议渲染进提示词 + 解析模型文本输出"，避免工具能力整段失效。

方言语义：
- ``native_openai``：发 ``tools`` 字段，解析 ``delta.tool_calls``；另常驻 DSML 兜底，
  因为无法探知服务方是否配置了工具解析器（配置缺失时模型会直接吐 DSML 文本）
- ``native_anthropic``：发 Anthropic tools，解析 ``tool_use`` 块
- ``dsml``：DeepSeek V4 原生 DSML 文本格式——请求侧仍发 ``tools``（服务端聊天模板消费），
  响应侧解析 ``<｜DSML｜>invoke``
- ``text_json``：模型不支持 function calling——请求侧**不发** ``tools``，把工具清单渲染进
  提示词，响应侧解析 ``<tool_call>{json}</tool_call>``
- ``none``：不支持且无可用文本协议——不注入任何工具（避免冗余 token 与输出混乱）

**零冗余原则**：同一模型的工具清单只出现一次——原生/DSML 走 ``tools`` 字段（不再渲染进
提示词），文本方言走提示词（不再发 ``tools`` 字段）。
"""

from __future__ import annotations

import json
from typing import Any

from app.infrastructure.llm.common import PROTOCOL_ANTHROPIC

NATIVE_OPENAI = "native_openai"
NATIVE_ANTHROPIC = "native_anthropic"
DSML = "dsml"
TEXT_JSON = "text_json"
NONE = "none"

# 通过 tools 字段（而非提示词）传递工具清单的方言
NATIVE_TOOLS_DIALECTS = frozenset({NATIVE_OPENAI, NATIVE_ANTHROPIC, DSML})
# 需要把工具清单渲染进提示词的方言
PROMPT_DIALECTS = frozenset({TEXT_JSON})
# 文本化工具调用协议（响应侧需文本解析器）
TEXT_DIALECTS = frozenset({DSML, TEXT_JSON})

# DeepSeek V4 的原生工具格式即 DSML（vLLM 有 deepseek_v4 parser；服务方未启用时模型直接吐文本）
_DSML_FAMILIES = ("deepseek-v4", "deepseek_v4")
# 不支持 function calling 时可按文本协议注入的模型家族（子串匹配，大小写不敏感）；
# 未登记的家族不注入（避免对无法遵循协议的模型注入工具，既浪费 token 又扰乱输出）
_TEXT_FAMILIES = ("hermes", "glm", "qwen", "llama", "mistral")


def resolve_tool_dialect(
    protocol_type: str,
    model_id: str,
    model_type: str = "chat",
    supports_tool_call: int = 0,
) -> str:
    """按协议类型 + 模型家族 + 能力位推导工具调用方言。

    非 chat 模型（embedding/rerank/image_* 等）不参与对话工具调用，返回 none。
    """
    if protocol_type == PROTOCOL_ANTHROPIC:
        return NATIVE_ANTHROPIC
    if model_type != "chat":
        return NONE

    lowered = (model_id or "").lower()
    if any(family in lowered for family in _DSML_FAMILIES):
        return DSML
    if supports_tool_call:
        return NATIVE_OPENAI
    if any(family in lowered for family in _TEXT_FAMILIES):
        return TEXT_JSON
    return NONE


def is_prompt_tool_family(model_id: str, model_type: str = "chat") -> bool:
    """该模型是否命中文本工具协议家族（不支持 function calling 时仍可按提示词注入工具）。

    供能力校验/降级链过滤使用：命中即视为"可用工具"，无需 ``tools`` 字段能力。
    """
    if model_type != "chat":
        return False
    lowered = (model_id or "").lower()
    return any(family in lowered for family in _TEXT_FAMILIES)


def render_tool_prompt(tools: list[dict[str, Any]] | None, dialect: str) -> str:
    """把工具清单渲染为文本协议说明（仅文本方言需要，返回值插入 system 消息）。

    原生方言返回空串——其工具清单经 ``tools`` 字段传递，再渲染即冗余。
    """
    if dialect not in PROMPT_DIALECTS or not tools:
        return ""
    lines = [
        "你可以调用以下工具检索信息或执行操作。",
        "调用方式：需要工具时，只输出下式内容（可连续多个，不要输出任何解释文字）：",
        '<tool_call>{"name":"工具名","arguments":{参数对象}}</tool_call>',
        "",
        "可用工具：",
    ]
    for spec in tools:
        function = spec.get("function") if isinstance(spec, dict) else None
        if not isinstance(function, dict) or not function.get("name"):
            continue
        description = (function.get("description") or "").strip()
        lines.append(f"- {function['name']}：{description}")
        parameters = function.get("parameters")
        if parameters:
            lines.append("  parameters: " + json.dumps(parameters, ensure_ascii=False))
    return "\n".join(lines)
