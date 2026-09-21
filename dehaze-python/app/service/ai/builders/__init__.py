"""图/工具/上下文构建：deepagents 图组装、团队图组装、业务工具装载。"""

from typing import TYPE_CHECKING

# 仅类型检查期可见：运行时不做聚合导入（避免包级循环 import），
# 但需让 pyright 认可 __all__ 中列出的子模块白名单。
if TYPE_CHECKING:
    from . import (
        context_manager,
        deep_agent_builder,
        dehaze_tools_builder,
        knowledge_base_tool,
        team_builder,
    )

__all__ = [
    "context_manager",
    "deep_agent_builder",
    "dehaze_tools_builder",
    "knowledge_base_tool",
    "team_builder",
]
