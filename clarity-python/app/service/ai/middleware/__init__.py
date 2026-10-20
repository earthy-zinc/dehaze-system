"""推理链横切：生命周期钩子、安全护栏、工具恢复、中断点管理。"""

from typing import TYPE_CHECKING

# 仅类型检查期可见：运行时不做聚合导入（避免包级循环 import），
# 但需让 pyright 认可 __all__ 中列出的子模块白名单。
if TYPE_CHECKING:
    from . import (
        agent_hooks,
        async_resume,
        capability_constraints,
        dehaze_hooks_middleware,
        guardrail_middleware,
        interrupt_handler,
        mcp_namespace_prefilter,
        paradigm_middleware,
        tool_failure_guard,
        tool_recovery,
    )

__all__ = [
    "agent_hooks",
    "async_resume",
    "capability_constraints",
    "dehaze_hooks_middleware",
    "guardrail_middleware",
    "interrupt_handler",
    "mcp_namespace_prefilter",
    "paradigm_middleware",
    "tool_failure_guard",
    "tool_recovery",
]
