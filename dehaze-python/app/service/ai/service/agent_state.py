"""Agent 推理状态定义

AgentState 为 LangGraph StateGraph 的节点间共享状态字典（TypedDict 描述）。
total=False 表示所有字段均可选，允许状态在推理过程中渐进式填充。
"""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # 上下文
    messages: list[dict[str, Any]]  # 对话消息列表 [{role, content}]
    user_id: int
    conversation_id: int
    message_id: int  # assistant 消息 ID
    model_id: str
    system_prompt: str | None
    stream_session_id: str  # SSE 流 ID

    # 推理控制
    complexity: str  # L0/L1/L2/L3
    reasoning_mode: str  # direct/react/plan_execute/reflexion
    step_count: int
    max_steps: int
    token_used: int
    token_budget: int
    thoughts: list[dict[str, Any]]  # 推理步骤记录
    tool_calls: list[dict[str, Any]]  # 本轮 LLM 决策的工具调用列表（OpenAI tool_call 格式）
    assistant_text: str  # 本轮 LLM 的思考文本（text_delta 累积，用于注入 assistant 消息 content）

    # 输出
    final_response: str
    stop_reason: str  # stop/tool_calls/length/content_filter/canceled/max_steps
    usage: dict[str, Any]  # token 统计
    error: str | None
    skill_instruction: str  # 已加载的 Skill 完整指令

    # 计费（before_agent 预扣后注入，贯穿滚动预算与实扣结算）
    billing_context: dict[
        str, Any
    ]  # {user_id, conversation_id, message_id, estimated_credits, budget_pool,
    # remaining_budget, billing_id}

    # 任务状态（不进入对话流，LLM 通过 get_task_status 工具查询）
    task_type: str  # 当前任务类型（如 dehaze/evaluate）
    task_algorithm: str  # 当前使用的算法标识
    task_params: dict[str, Any]  # 处理参数（如强度、饱和度）
    task_status: str  # 任务执行状态（processing/completed/failed）
    task_id: str  # 关联的异步任务 ID
    task_artifacts: list[dict[str, Any]]  # 任务产生的产物引用列表（ID + 类型 + 摘要）
