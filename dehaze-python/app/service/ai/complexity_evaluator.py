"""复杂度评估器

两级评估：规则预判（不调 LLM）优先，规则无法判定时调用 LLM 兜底评估。
评估结果决定推理范式：L0→direct、L1→react、L2→plan_execute、L3→reflexion。
"""

from app.database import get_db_session
from app.dependencies.redis import get_redis_client
from app.service.ai.agent_state import AgentState
from app.infrastructure.llm.llm_client import llm_client

# 动作关键词 → L1（ReAct）
_ACTION_KEYWORDS = ("去雾", "处理", "评估", "分析")
# 批量关键词 → L2（Plan-and-Execute）
_BATCH_KEYWORDS = ("批量", "所有", "这些图片", "批量处理")
# 高精度关键词 → L3（Reflexion）
_PRECISION_KEYWORDS = ("报告", "审查", "优化", "确保符合")

# LLM 兜底评估提示词
_EVAL_PROMPT = (
    "请评估用户消息的任务复杂度，只返回一个等级："
    "L0(简单问答，无需工具)、L1(单步决策，可能需要工具)、"
    "L2(多步骤批量任务)、L3(高精度需迭代优化的任务)。"
)


def _rule_based_eval(content: str) -> str | None:
    """规则预判复杂度，无法判定返回 None"""
    if any(k in content for k in _PRECISION_KEYWORDS):
        return "L3"
    if any(k in content for k in _BATCH_KEYWORDS):
        return "L2"
    if any(k in content for k in _ACTION_KEYWORDS):
        return "L1"
    if len(content) < 50:
        return "L0"
    return None


def _get_last_content(messages: list[dict]) -> str:
    """从消息列表提取最后一条消息的文本内容"""
    if not messages:
        return ""
    return messages[-1].get("content", "")


async def _llm_eval(state: AgentState) -> tuple[str, dict]:
    """LLM 兜底评估复杂度，失败时保守返回 L0。返回 (等级, usage)"""
    usage: dict = {}
    try:
        last_content = _get_last_content(state["messages"])
        async with get_db_session() as db:
            redis = await get_redis_client()
            content = ""
            async for chunk in llm_client.stream_chat(
                db,
                redis,
                state["model_id"],
                [{"role": "user", "content": last_content}],
                system_prompt=_EVAL_PROMPT,
                temperature=0,
                max_tokens=10,
            ):
                if chunk.type == "text_delta":
                    content += chunk.content
                elif chunk.type == "done":
                    usage = chunk.usage or {}
        for level in ("L0", "L1", "L2", "L3"):
            if level in content.upper():
                return level, usage
    except Exception:
        pass
    return "L0", usage


async def evaluate_complexity(state: AgentState) -> dict:
    """评估复杂度，返回 {complexity, reasoning_mode, usage}"""
    complexity = _rule_based_eval(_get_last_content(state["messages"]))
    usage: dict = {}
    if complexity is None:
        complexity, usage = await _llm_eval(state)
    mode = {"L0": "direct", "L1": "react", "L2": "plan_execute", "L3": "reflexion"}.get(
        complexity, "direct"
    )
    return {"complexity": complexity, "reasoning_mode": mode, "usage": usage}
