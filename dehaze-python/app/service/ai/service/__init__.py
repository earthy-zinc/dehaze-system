"""业务编排层：推理用例 + 领域服务。

外部引用统一走模块路径：`from app.service.ai.service.<module> import <symbol>`。
子包模块间存在跨子包依赖环，故不做聚合导入（避免包级循环 import），
`__all__` 即本子包对外暴露的模块白名单。
"""

from typing import TYPE_CHECKING

# 仅类型检查期可见：运行时不做聚合导入（见模块 docstring），
# 但需让 pyright 认可 __all__ 中列出的子模块白名单。
if TYPE_CHECKING:
    from . import (
        agent_state,
        ai_schedule_executor,
        ai_schedule_notify,
        ai_schedule_service,
        algorithm_recommend_service,
        batch_process_service,
        compatible_api_service,
        compatible_audit,
        compatible_governance,
        conversation_search_service,
        credits_service,
        eval_runner,
        memory_es_service,
        memory_extraction,
        memory_injection,
        provider_connectivity_service,
        reasoning_service,
        skill_manager,
        step_summarizer,
        suggestion_service,
        summary_service,
    )

__all__ = [
    "agent_state",
    "ai_schedule_executor",
    "ai_schedule_notify",
    "ai_schedule_service",
    "algorithm_recommend_service",
    "batch_process_service",
    "compatible_api_service",
    "compatible_audit",
    "compatible_governance",
    "conversation_search_service",
    "credits_service",
    "eval_runner",
    "memory_es_service",
    "memory_extraction",
    "memory_injection",
    "provider_connectivity_service",
    "reasoning_service",
    "skill_manager",
    "step_summarizer",
    "suggestion_service",
    "summary_service",
]
