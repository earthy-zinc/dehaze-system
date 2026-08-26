"""MCP 工具命名空间预筛选

设计文档 §3（能力扩展）before_model 钩子职责：每次模型调用前按用户意图预筛选
MCP 工具的加载范围，避免全量注入数万个 token 的工具定义挤占上下文。

机制（四步）：
1. list_namespaces()：调 MCP 网关 tools/list 获取命名空间摘要，缓存 5 分钟；
   网关不可用返回空列表（降级：仅靠 mcp_lookup_tool 元工具）。
2. match_namespaces()：规则预判（不调 LLM），按关键词命中 1-2 个命名空间。
3. expand()：命中时展开完整工具定义；未命中注入摘要 + mcp_lookup_tool 使用指引。
4. build_tools_block()：组装注入文本。

注入位置：动态工具块不能进入图缓存的 system_prompt（图按 Agent 版本缓存复用），
必须在模型调用时经消息层注入 —— 由 McpNamespacePrefilterMiddleware.awrap_model_call
追加到本次 system message。
"""

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import SystemMessage

from app.config import settings
from app.infrastructure.clients.mcp_gateway_client import mcp_gateway_client

logger = logging.getLogger(__name__)

# 元 tool 名：网关自身能力，不参与命名空间分组
_META_TOOLS = frozenset({"lookup_tool", "lookup_tool_param_schema", "execute_tool"})

# 命名空间关键词映射（Step2 规则预判）。值为中英文关键词，命中任一即匹配该命名空间。
NAMESPACE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "image_processing": (
        "去雾",
        "处理",
        "增强",
        "图像",
        "图片",
        "defog",
        "dehaze",
        "enhance",
        "image",
    ),
    "evaluation": (
        "评估",
        "PSNR",
        "SSIM",
        "指标",
        "评价",
        "quality",
        "metric",
        "evaluate",
        "psnr",
        "ssim",
    ),
    "algorithm": ("算法", "推荐", "选型", "model", "algorithm", "recommend"),
    "batch": ("批量", "批处理", "并发", "batch", "parallel"),
    "preset": ("预设", "模板", "preset", "template"),
    "model": ("模型", "权重", "下载", "model"),
    "dataset": ("数据集", "数据", "dataset", "data"),
    "user": ("用户", "账号", "资料", "user", "profile", "account"),
}

# 命名空间默认描述（网关未提供分组描述时的兜底）
_NAMESPACE_DESCRIPTIONS: dict[str, str] = {
    "image_processing": "图像去雾/处理/增强相关 API",
    "evaluation": "图像质量评估（PSNR/SSIM）相关 API",
    "algorithm": "算法选择与推荐相关 API",
    "batch": "批量处理相关 API",
    "preset": "预设与模板相关 API",
    "model": "模型资源相关 API",
    "dataset": "数据集管理相关 API",
    "user": "用户与账号相关 API",
}


def _namespace_of(tool_name: str) -> str | None:
    """推导工具所属命名空间：按已知命名空间做最长前缀匹配。

    命名空间名可为多段（如 image_processing），工具命名为 <namespace>_<tool>
    （对齐 Guardrail 越权校验约定），故按命名空间键长度降序匹配前缀。
    """
    if tool_name in _META_TOOLS or not tool_name:
        return None
    for ns in sorted(NAMESPACE_KEYWORDS, key=len, reverse=True):
        if tool_name.startswith(ns + "_"):
            return ns
    return None


def _tool_to_text(tool: dict[str, Any]) -> str:
    """将单个工具定义格式化为注入文本。

    优先展示 expand 获取的完整参数定义（schema_text，来自网关
    lookup_tool_param_schema）；无展开文本时退化为参数名列表。
    """
    name = tool.get("name", "")
    desc = tool.get("description", "")
    schema_text = tool.get("schema_text") or ""
    if schema_text:
        return f"- `{name}`：{desc or '无描述'}\n  参数定义：{schema_text}"
    props = (tool.get("input_schema") or {}).get("properties") or {}
    params = "、".join(props.keys()) if props else "无参数"
    return f"- `{name}`：{desc or '无描述'}（参数：{params}）"


class McpNamespacePrefilter:
    """MCP 命名空间预筛选：摘要缓存 + 意图匹配 + 工具块组装。

    与网关的耦合仅通过 gateway 注入（默认 mcp_gateway_client 单例），便于单测 mock。
    """

    def __init__(self, gateway: Any = None) -> None:
        self.gateway = gateway if gateway is not None else mcp_gateway_client
        # 摘要缓存：{"fetched_at": float, "namespaces": {ns: {name, description, tool_count}}}
        self._cache: dict[str, Any] = {}

    def _cache_fresh(self) -> bool:
        return bool(self._cache) and (
            time.monotonic() - self._cache["fetched_at"] < settings.AI_MCP_NAMESPACE_CACHE_TTL
        )

    async def list_namespaces(self) -> dict[str, dict[str, Any]]:
        """获取命名空间摘要（name + description + tool_count），缓存 5 分钟。

        网关不可用/无业务工具时返回空 dict（降级：仅靠 mcp_lookup_tool 元工具）。
        """
        if self._cache_fresh():
            return self._cache["namespaces"]
        tools = await self.gateway.list_tools()
        namespaces: dict[str, dict[str, Any]] = {}
        for t in tools:
            ns = _namespace_of(t.get("name", ""))
            if not ns:
                continue
            entry = namespaces.setdefault(
                ns,
                {
                    "name": ns,
                    "description": _NAMESPACE_DESCRIPTIONS.get(ns, ""),
                    "tool_count": 0,
                    "tools": [],
                },
            )
            entry["tool_count"] += 1
            entry["tools"].append(t)
        if not tools:
            logger.warning("MCP 网关 tools/list 返回空，命名空间预筛选降级为仅 mcp_lookup_tool")
        self._cache = {"fetched_at": time.monotonic(), "namespaces": namespaces}
        return namespaces

    def match_namespaces(self, user_text: str) -> list[str]:
        """规则预判：按关键词匹配 1-2 个命名空间，无法匹配返回空列表。"""
        if not user_text:
            return []
        text = user_text.lower()
        matched: list[str] = []
        for ns, keywords in NAMESPACE_KEYWORDS.items():
            if any(kw.lower() in text for kw in keywords):
                matched.append(ns)
            if len(matched) >= 2:
                break
        return matched

    async def expand(self, namespaces: list[str]) -> list[dict[str, Any]]:
        """展开匹配命名空间的完整工具定义。

        MCP SDK 的 list_tools 不支持 expand 参数，故对命中命名空间内的工具逐个调
        lookup_tool_param_schema 获取完整参数定义；网关元 tool 为普通业务工具不参与。
        无工具定义可展开时返回空列表（上层走模糊兜底）。
        """
        if not namespaces:
            return []
        summaries = await self.list_namespaces()
        full: list[dict[str, Any]] = []
        for ns in namespaces:
            entry = summaries.get(ns)
            if not entry:
                continue
            for t in entry.get("tools") or []:
                tool_name = t.get("name", "")
                if not tool_name:
                    continue
                schema_text = await self.gateway.lookup_tool_param_schema(tool_name)
                full.append(
                    {
                        "name": tool_name,
                        "description": t.get("description", ""),
                        "input_schema": t.get("input_schema") or {},
                        "schema_text": schema_text or "",
                    }
                )
        return full

    async def build_tools_block(
        self, user_text: str, agent_namespaces: list[str] | None = None
    ) -> str:
        """组装注入文本：命中且在授权范围内 → 展开完整定义；未命中 → 摘要 + 使用指引。"""
        matched = self.match_namespaces(user_text)
        if agent_namespaces is not None:
            matched = [ns for ns in matched if ns in agent_namespaces]
        summaries = await self.list_namespaces()

        if matched and any(summaries.get(ns) for ns in matched):
            definitions = await self.expand(matched)
            if definitions:
                lines = [
                    "以下为本次任务相关的 MCP 工具定义，可直接按需调用：",
                    *[_tool_to_text(d) for d in definitions],
                ]
                return "\n".join(lines)

        # 模糊兜底：注入命名空间摘要 + mcp_lookup_tool 使用指引
        lines = ["以下是当前可用的 MCP 能力命名空间摘要："]
        if summaries:
            for ns in sorted(summaries):
                entry = summaries[ns]
                lines.append(
                    f"- {entry['name']}：{entry['description'] or '无描述'}"
                    f"（{entry['tool_count']} 个工具）"
                )
        else:
            lines.append("- （网关未返回可用的 MCP 命名空间）")
        lines.extend(
            [
                "",
                "如需查找具体工具，请调用 `mcp_lookup_tool` 元工具，传入关键词（如“去雾”）",
                "搜索后端 API，再通过 `mcp_execute_tool` 执行。",
            ]
        )
        return "\n".join(lines)


class McpNamespacePrefilterMiddleware(AgentMiddleware):
    """deepagents 中间件：在模型调用时注入预筛选的动态工具块。

    工具块随本次消息注入 system message（消息层），不进入图缓存键的 system_prompt，
    图实例按 Agent 版本缓存复用不受影响。异步执行不阻塞图构建。
    """

    def __init__(
        self,
        agent_namespaces: list[str] | None = None,
        prefilter: McpNamespacePrefilter | None = None,
    ) -> None:
        self.agent_namespaces = agent_namespaces
        self.prefilter = prefilter if prefilter is not None else McpNamespacePrefilter()

    async def awrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        user_text = self._last_user_text(request.state)
        if user_text:
            block = await self.prefilter.build_tools_block(user_text, self.agent_namespaces)
            if block:
                base = request.system_message.content if request.system_message else ""
                composed = f"{base}\n\n{block}" if base else block
                request = request.override(system_message=SystemMessage(content=composed))
        return await handler(request)

    @staticmethod
    def _last_user_text(state: Any) -> str:
        messages = state.get("messages") or []
        for m in reversed(messages):
            if getattr(m, "type", "") == "human" and getattr(m, "content", None):
                return str(m.content)
        return ""
