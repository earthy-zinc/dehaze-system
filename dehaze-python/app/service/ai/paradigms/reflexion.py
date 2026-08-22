"""Reflexion 范式核心逻辑（纯逻辑，与 LLM 调用/DB 解耦）

设计文档 §4.2。流程：执行后 evaluator 自评 0-1 分（叠加规则校验）→ 分数低于
reflexion_threshold 时 self_reflection 分析根因并生成改进策略 → 写入反思记忆
（source=reflection）→ 下一轮注入 → 超过 max_iterations_reflexion 后接受当前最佳。

所有 LLM 能力经注入的 `model_call(messages, system_prompt) -> str` 完成；
反思记忆经注入的 `save_memory(**kwargs)` 落库，便于单测 mock。
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

# evaluator 提示词：要求 LLM 按评估维度给 0-1 分
_EVALUATOR_PROMPT = """你是质量评审。根据任务要求评估给定输出，仅返回 JSON（不要额外文字）：
{"score": 0-1之间的小数, "feedback": "简要评估说明"}

评估维度：是否满足任务要求、是否完整无遗漏、是否存在事实/逻辑错误、是否可直接使用。
"""

# self_reflection 提示词：分析根因并生成改进策略
_REFLECTION_PROMPT = """你是反思者。上轮执行未达要求，分析失败根因并给出具体改进策略。
仅返回 JSON（不要额外文字）：
{"root_cause": "根因分析", "strategy": "下一轮执行的改进策略"}

上轮评估反馈：{feedback}
"""

# 反思记忆写入源
REFLECTION_MEMORY_SOURCE = "reflection"

# 期望输出格式的规则校验（expected 非空时启用）：
# 仅做最基础的格式存在性检查，深层次语义判断交给 LLM evaluator。
_FORMAT_CHECKS = {
    "json": lambda s: _looks_like_json(s),
    "list": lambda s: bool(re.search(r"[\n\-\*]|\d+[\.\、\)]", s)),
}


def _looks_like_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except Exception:
        return (s.strip().startswith("{") and s.strip().endswith("}")) or (
            s.strip().startswith("[") and s.strip().endswith("]")
        )


async def evaluate_output(
    requirement: str,
    output: str,
    model_call: Callable[[list[dict], str], Awaitable[str]],
    expected: str | None = None,
) -> tuple[float, str]:
    """执行后自评输出质量，返回 (score, feedback)。

    LLM 自评 0-1 分；若配置了 expected 格式，叠加规则校验：格式明显不满足时
    直接压分（score 减半），保证规则兜底与 LLM 自评不冲突。
    """
    score = 0.0
    feedback = ""
    try:
        raw = await model_call(
            [{"role": "user", "content": f"任务要求：{requirement}\n\n输出：\n{output}"}],
            _EVALUATOR_PROMPT,
        )
        data = _parse_score_json(raw)
        score = max(0.0, min(1.0, float(data.get("score") or 0.0)))
        feedback = str(data.get("feedback") or "")
    except Exception:
        # 自评失败保守给低分，让 reflection 有机会介入
        score = 0.0
        feedback = "评估器解析失败，按不达标处理"

    if expected:
        check = _FORMAT_CHECKS.get(expected)
        if check and not check(output or ""):
            score *= 0.5
            feedback = f"输出不符合期望格式[{expected}]。{feedback}".strip()
    return score, feedback


async def reflect_failure(
    requirement: str,
    output: str,
    feedback: str,
    model_call: Callable[[list[dict], str], Awaitable[str]],
) -> dict[str, Any]:
    """self_reflection：分析根因并生成改进策略，返回 {root_cause, strategy}。"""
    try:
        raw = await model_call(
            [
                {
                    "role": "user",
                    "content": f"任务要求：{requirement}\n\n当前输出：\n{output}",
                }
            ],
            _REFLECTION_PROMPT.format(feedback=feedback or ""),
        )
        data = _parse_score_json(raw)
        return {
            "root_cause": str(data.get("root_cause") or "未知"),
            "strategy": str(data.get("strategy") or ""),
        }
    except Exception:
        return {"root_cause": "评估失败", "strategy": "重试一次"}


def _parse_score_json(raw: str) -> dict:
    """从 LLM 输出提取 JSON 对象（容忍代码块/杂文包裹）。"""
    text = (raw or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM 输出不含 JSON 对象")
    return json.loads(text[start : end + 1])


def reflexion_loop(
    *,
    run_actor: Callable[[list[dict], str], Awaitable[str]],
    evaluate: Callable[..., Awaitable[tuple[float, str]]],
    reflect: Callable[..., Awaitable[dict[str, Any]]],
    max_iterations: int,
    threshold: float,
) -> Callable[..., Awaitable[tuple[str, list[dict[str, Any]]]]]:
    """Reflexion 迭代编排（可单测、可复用于 ReAct/子任务执行）。

    返回 async 函数，签名与 evaluate/reflect 对齐：接受 (requirement, messages)，
    返回 (best_output, rounds)；rounds 记录每轮 {score, feedback, strategy}。

    超过 max_iterations 后接受当前最佳（最高分轮次输出）。
    """

    async def _run(requirement: str, messages: list[dict]) -> tuple[str, list[dict[str, Any]]]:
        rounds: list[dict[str, Any]] = []
        best_score = -1.0
        best_output = ""
        reflection = None
        for _ in range(max(1, max_iterations)):
            prompt = f"任务要求：{requirement}"
            if reflection:
                prompt += f"\n\n历史反思（应避免重蹈覆辙）：{reflection.get('strategy')}"
            output = await run_actor(messages, prompt)
            score, feedback = await evaluate(requirement, output)
            rounds.append({"score": score, "feedback": feedback, "output": output})
            if score > best_score:
                best_score = score
                best_output = output
            if score >= threshold:
                return best_output, rounds
            reflection = await reflect(requirement, output, feedback)
            rounds[-1]["strategy"] = reflection.get("strategy", "")
        return best_output, rounds

    return _run


def build_reflection_memory(
    user_id: int,
    conversation_id: int | None,
    model_id: str,
    requirement: str,
    reflection: dict[str, Any],
    skill: str | None = None,
) -> dict[str, Any]:
    """构造反思记忆实体字段（source=reflection）。

    与 memory_extraction 保存路径共用 sys_ai_memory 表；source 用 reflection 标识，
    由 memory_injection 检索层按检索命中注入（同任务再执行时复用历史反思经验）。
    """
    memory: dict[str, Any] = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "model_id": model_id,
        "content": (
            f"任务要求：{requirement}\n根因：{reflection.get('root_cause')}\n"
            f"改进策略：{reflection.get('strategy')}"
        ),
        "source": REFLECTION_MEMORY_SOURCE,
        "source_type": "self_reflection",
    }
    if skill:
        memory["metadata"] = {"skill": skill}
    return memory
