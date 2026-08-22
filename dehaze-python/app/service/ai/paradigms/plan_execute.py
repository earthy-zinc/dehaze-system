"""Plan-and-Execute 范式核心逻辑（纯逻辑，与 LLM 调用解耦）

设计文档 §4.1/§4.2/§4.3。本模块只负责"计划如何被理解、划分、修订、干预"，
不直接调用 LLM：所有 LLM 能力经注入的 `model_call(messages, system_prompt) -> str`
完成，便于单测 mock 与真实 `DehazeChatModel` 复用。

计划结构（写入 state["plan"]）：
    {
        "tasks": [PlanTask],   # 扁平任务列表
        "status": "pending|executing|revised|done",
        "revisions": [...],    # Replanner 修订记录
    }
PlanTask: {id, description, depends_on[], tool_hint?, paradigm?, status, result?}
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from typing import Any

# 子任务默认范式：未标注时按 react（单步 ReAct）执行
DEFAULT_SUBTASK_PARADIGM = "react"

# Planner 提示词：要求 LLM 输出结构化 JSON 计划
_PLANNER_PROMPT = """你是任务分解引擎。将用户任务分解为可并行/串行的子任务列表，
仅返回 JSON（不要任何额外文字），结构如下：
{"tasks": [{"id": "A", "description": "子任务说明", "depends_on": ["依赖任务id"],
"tool_hint": "可选，提示可用工具名", "paradigm": "可选，react|reflexion"}]}

要求：
- id 用大写字母（A/B/C...），depends_on 声明前置依赖，无依赖则留空数组
- 相互独立的子任务不要互相依赖（它们会并行执行）
- 涉及高精度/需复核的子任务 paradigm 标注 reflexion，其余省略或 react
"""

# Replanner 提示词：只重规划受影响子任务，不重跑全计划
_REPLANNER_PROMPT = """你是计划修订引擎。给定执行中失败/异常的子任务与当前完整计划，
仅重新规划受影响的子任务。只返回 JSON（不要额外文字）：
{"revised": [{"id": "新id", "description": "修订后说明", "depends_on": [], "tool_hint": null}]}
其中 revised 是与受影响子任务等价的修订集合（可拆分/合并/调整顺序），
未受影响子任务不要出现在 revised 中。
"""


# ── 数据模型 ──────────────────────────────────────────


def new_plan() -> dict[str, Any]:
    """构造空计划。"""
    return {"tasks": [], "status": "pending", "revisions": []}


def _new_task(
    task_id: str,
    description: str,
    depends_on: list[str] | None = None,
    tool_hint: str | None = None,
    paradigm: str | None = None,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "description": description,
        "depends_on": list(depends_on or []),
        "tool_hint": tool_hint,
        "paradigm": paradigm or DEFAULT_SUBTASK_PARADIGM,
        "status": "pending",
        "result": None,
    }


# ── Planner ──────────────────────────────────────────


async def build_plan(
    task: str,
    model_call: Callable[[list[dict], str], Awaitable[str]],
    tool_hint: str | None = None,
) -> dict[str, Any]:
    """调用 LLM 将任务分解为结构化计划。

    返回 {tasks: [PlanTask], status, revisions}。解析失败时兜底为单任务 react 计划，
    保证计划始终可执行（宁可退化为单步，也不中断推理）。
    """
    plan = new_plan()
    user_prompt = f"用户任务：{task}\n工具提示：{tool_hint or '无'}"
    try:
        raw = await model_call([{"role": "user", "content": user_prompt}], _PLANNER_PROMPT)
        data = _extract_json(raw)
        tasks = []
        for i, item in enumerate(data.get("tasks") or [], start=1):
            tasks.append(
                _new_task(
                    task_id=str(item.get("id") or _auto_id(i)),
                    description=str(item.get("description") or "").strip(),
                    depends_on=[str(d) for d in (item.get("depends_on") or [])],
                    tool_hint=(item.get("tool_hint") or None),
                    paradigm=(item.get("paradigm") or None),
                )
            )
        if tasks:
            plan["tasks"] = tasks
    except Exception:
        # 解析失败：退化为单任务 react 计划，不中断推理
        plan["tasks"] = [_new_task("A", task, [], tool_hint, "react")]
    return plan


# ── 依赖拓扑分批 ──────────────────────────────────────


def compute_batches(tasks: list[dict[str, Any]]) -> list[list[str]]:
    """按依赖拓扑把任务分成可并行执行的批。

    返回 list[list[task_id]]：同批内任务无相互依赖可并行，批间严格串行。
    环依赖按先到先得顺序拆批避免死循环。
    """
    by_id = {t["id"]: t for t in tasks}
    # 计算每个任务的剩余依赖数与反向依赖
    remaining = {tid: set(by_id[tid].get("depends_on") or []) for tid in by_id}
    dependents: dict[str, set[str]] = defaultdict(set)
    for tid in by_id:
        for dep in remaining[tid]:
            dependents[dep].add(tid)

    batches: list[list[str]] = []
    done: set[str] = set()
    while len(done) < len(by_id):
        # 就绪队列：无未满足依赖（或依赖缺失）的任务；无就绪时从被卡住任务中
        # 取一个（环依赖兜底），保证每次循环都有进展、不陷入死循环
        ready = deque(tid for tid in by_id if tid not in done and not remaining[tid])
        if not ready:
            stuck = next((tid for tid in by_id if tid not in done), None)
            if stuck is None:
                break
            ready.append(stuck)
        batch = []
        for _ in range(len(ready)):
            tid = ready.popleft()
            if tid in done:
                continue
            done.add(tid)
            batch.append(tid)
            for nxt in dependents[tid]:
                if nxt in done:
                    continue
                remaining[nxt].discard(tid)
                if not remaining[nxt]:
                    ready.append(nxt)
        if batch:
            batches.append(batch)
    return batches


# ── Replanner ─────────────────────────────────────────


async def replan(
    plan: dict[str, Any],
    failed_task_ids: list[str],
    model_call: Callable[[list[dict], str], Awaitable[str]],
) -> dict[str, Any]:
    """子任务失败/异常时重新规划受影响部分。

    只修订失败相关任务（可拆分/合并/调整），未受影响任务原样保留，
    修订记录追加到 plan["revisions"]，状态置 revised。
    """
    if not failed_task_ids:
        return plan
    reason = "，".join(failed_task_ids)
    context = json.dumps(
        [
            {"id": t["id"], "description": t["description"], "depends_on": t.get("depends_on")}
            for t in plan.get("tasks") or []
        ],
        ensure_ascii=False,
    )
    try:
        raw = await model_call(
            [{"role": "user", "content": f"失败子任务：{reason}\n当前计划：{context}"}],
            _REPLANNER_PROMPT,
        )
        data = _extract_json(raw)
        revised = []
        for i, item in enumerate(data.get("revised") or [], start=1):
            revised.append(
                _new_task(
                    task_id=str(item.get("id") or _auto_id(i)),
                    description=str(item.get("description") or "").strip(),
                    depends_on=[str(d) for d in (item.get("depends_on") or [])],
                    tool_hint=(item.get("tool_hint") or None),
                    paradigm=(item.get("paradigm") or None),
                )
            )
        if revised:
            _apply_revision(plan, failed_task_ids, revised)
    except Exception:
        # 修订失败：把失败任务直接标注失败，避免无限重规划
        for t in plan.get("tasks") or []:
            if t["id"] in set(failed_task_ids):
                t["status"] = "failed"
    plan["status"] = "revised"
    return plan


def _apply_revision(
    plan: dict[str, Any], failed_task_ids: list[str], revised: list[dict[str, Any]]
) -> None:
    """把修订集合替换掉失败任务，追加修订记录。"""
    failed_set = set(failed_task_ids)
    kept = [t for t in plan.get("tasks") or [] if t["id"] not in failed_set]
    plan["tasks"] = kept + revised
    plan["revisions"].append(
        {
            "at": len(plan["revisions"]) + 1,
            "reason": "，".join(failed_task_ids),
            "change": json.dumps([t["id"] for t in revised], ensure_ascii=False),
        }
    )


# ── 计划干预（plan_edit）───────────────────────────────


def apply_plan_edit(plan: dict[str, Any], plan_edit: dict[str, Any] | None) -> dict[str, Any]:
    """合并用户计划干预（resume 透传），并做干预窗口校验。

    plan_edit 结构：{remove: [taskId], reorder: [taskId...], add: {description, depends_on}}

    干预窗口：仅 plan.status == "pending" 时允许整体调整（remove/reorder/add）；
    executing 之后不允许调整（中途改需求应走新消息）。
    """
    if not plan_edit:
        return plan
    if plan.get("status") not in ("pending", "revised"):
        raise ValueError("计划已开始执行，无法整体干预；如需修改需求请发送新消息")
    tasks = plan.get("tasks") or []
    # 所有原始任务 id（含将被移除的）都占用，避免新增任务复用刚释放的 id 造成混淆
    used_ids = {t["id"] for t in tasks}

    # remove：删除指定任务及其被依赖关系
    remove_set = set(plan_edit.get("remove") or [])
    kept = [t for t in tasks if t["id"] not in remove_set]
    for t in kept:
        t["depends_on"] = [d for d in t.get("depends_on") or [] if d not in remove_set]

    # add：追加新任务
    add_spec = plan_edit.get("add")
    if isinstance(add_spec, dict) and add_spec.get("description"):
        new_id = _unused_id(used_ids)
        kept.append(
            _new_task(
                new_id,
                str(add_spec["description"]),
                [str(d) for d in (add_spec.get("depends_on") or [])],
                add_spec.get("tool_hint"),
                add_spec.get("paradigm"),
            )
        )

    # reorder：按指定顺序重排（未列举的任务保持相对顺序追加其后）
    if plan_edit.get("reorder"):
        order = [str(x) for x in plan_edit["reorder"]]
        indexed = {t["id"]: t for t in kept}
        reordered = [indexed[oid] for oid in order if oid in indexed]
        rest = [t for t in kept if t["id"] not in set(order)]
        kept = reordered + rest

    plan["tasks"] = kept
    plan["status"] = "pending"
    return plan


# ── 内部工具 ──────────────────────────────────────────


def _auto_id(index: int) -> str:
    """生成任务 id：A/Z 之后用 A1/Z1 命名空间。"""
    if index <= 26:
        return chr(ord("A") + index - 1)
    return f"{chr(ord('A') + (index - 1) % 26)}{(index - 1) // 26}"


def _unused_id(existing: set[str]) -> str:
    i = 1
    while _auto_id(i) in existing:
        i += 1
    return _auto_id(i)


def _extract_json(raw: str) -> dict:
    """从 LLM 输出中提取 JSON 对象（容忍代码块/前后杂文包裹）。"""
    text = (raw or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM 输出不含 JSON 对象")
    return json.loads(text[start : end + 1])
