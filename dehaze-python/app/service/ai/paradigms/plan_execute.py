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
import logging
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Collection
from typing import Any

logger = logging.getLogger(__name__)

# 子任务默认范式：未标注时按 react（单步 ReAct）执行
DEFAULT_SUBTASK_PARADIGM = "react"

# tool_hint 白名单别名表（自由文本 → 真实工具名）。
# PRJ-1：Planner 由 LLM 生成，tool_hint 常是「知识库检索」「none」这类自然语言，
# 直接透传会让计划承诺做不到的事；这里只做收敛，命中不了就丢弃。
_TOOL_HINT_ALIASES: dict[str, str] = {
    "知识库检索": "knowledge_base_search",
    "知识库搜索": "knowledge_base_search",
    "知识库": "knowledge_base_search",
    "knowledge_retrieval": "knowledge_base_search",
    "kb_search": "knowledge_base_search",
    "knowledge_base_search": "knowledge_base_search",
    "联网搜索": "web_search",
    "网络搜索": "web_search",
    "web_search": "web_search",
    "任务状态": "get_task_status",
    "任务查询": "get_task_status",
    "get_task_status": "get_task_status",
    "算法推荐": "algorithm_recommend",
    "algorithm_recommend": "algorithm_recommend",
    "批量处理": "batch_process",
    "batch_process": "batch_process",
    "代码执行": "execute_code",
    "execute_code": "execute_code",
    "读取产物": "visual_read",
    "visual_read": "visual_read",
}

# 明显不是工具名的字面量：计划里常见 "none"/"无"，一律视为未提示
_NON_TOOL_HINTS = frozenset({"none", "null", "n/a", "na", "无", "没有", "-", "无工具"})


def resolve_tool_hint(hint: str | None, available_tools: Collection[str] | None) -> str | None:
    """把计划里的自由文本 tool_hint 归一为真实工具名；无法确证时返回 None。

    收敛规则（宁可不提示，也不误导）：
    - 空值 / none / 无 等非工具字面量一律丢弃；
    - 给定 available_tools 时严格白名单：与真实工具名一致、或别名命中且在可用集合内才保留；
    - available_tools 为 None（调用方未提供白名单）时退化为 best-effort：别名命中则归一，
      否则按旧契约原样保留（生产链路由 ParadigmMiddleware 始终传入工具名列表）。
    """
    if hint is None:
        return None
    raw = str(hint).strip()
    if not raw or raw.lower() in _NON_TOOL_HINTS:
        return None
    mapped = _TOOL_HINT_ALIASES.get(raw) or _TOOL_HINT_ALIASES.get(raw.lower())
    if available_tools is None:
        return mapped or raw
    known = set(available_tools)
    if raw in known:
        return raw
    return mapped if mapped in known else None


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


def to_wire_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """把内部计划归一为对外 camelCase 形状（plan 事件与 plan_approve 中断的唯一出口）。

    内部结构保持 snake_case（Redis 中断点与执行器直接消费），仅在 SSE 出口归一一次；
    可选键（paradigm/toolHint/result）缺失即不下发，不塞 None。
    """
    tasks = []
    for t in plan.get("tasks") or []:
        task = {
            "id": t.get("id"),
            "description": t.get("description"),
            "dependsOn": t.get("depends_on") or [],
            "status": t.get("status", "pending"),
        }
        if t.get("paradigm") is not None:
            task["paradigm"] = t["paradigm"]
        if t.get("tool_hint") is not None:
            task["toolHint"] = t["tool_hint"]
        if t.get("result") is not None:
            task["result"] = t["result"]
        tasks.append(task)
    return {
        "tasks": tasks,
        "status": plan.get("status", "pending"),
        "revisions": plan.get("revisions") or [],
    }


# ── Planner ──────────────────────────────────────────


async def build_plan(
    task: str,
    model_call: Callable[[list[dict], str], Awaitable[str]],
    tool_hint: str | None = None,
    available_tools: Collection[str] | None = None,
) -> dict[str, Any]:
    """调用 LLM 将任务分解为结构化计划。

    返回 {tasks: [PlanTask], status, revisions}。仅当 LLM 输出无法解析为合法计划
    （非 JSON / 结构非法）时降级为单任务 react 计划（宁可退化为单步也不中断推理）；
    该降级是文档化契约，但会记 warning 日志暴露，便于排查提示词/模型合规问题。
    模型调用本身的异常（含护栏拦截/配额中断）不在此吞掉，直接冒泡交给调用方处理。

    available_tools 为当前范式可用的真实工具名集合：会写进 Planner 提示词的
    「可用工具白名单」，并把每个任务的 tool_hint 收敛为白名单内的真实工具名或 None
    （PRJ-1：此前 tool_hint 是自由文本，计划会承诺做不到的工具）。
    """
    plan = new_plan()
    user_prompt = f"用户任务：{task}\n工具提示：{tool_hint or '无'}"
    raw = ""
    try:
        raw = await model_call(
            [{"role": "user", "content": user_prompt}], _planner_prompt(available_tools)
        )
        data = _extract_json(raw)
        tasks = []
        for i, item in enumerate(data.get("tasks") or [], start=1):
            tasks.append(
                _new_task(
                    task_id=str(item.get("id") or _auto_id(i)),
                    description=str(item.get("description") or "").strip(),
                    depends_on=[str(d) for d in (item.get("depends_on") or [])],
                    tool_hint=resolve_tool_hint(item.get("tool_hint"), available_tools),
                    paradigm=(item.get("paradigm") or None),
                )
            )
        if tasks:
            plan["tasks"] = tasks
    except (ValueError, AttributeError, TypeError) as exc:
        # 仅解析类失败（非 JSON / 结构非法）才降级为单任务，并记日志暴露 LLM 输出不合规
        logger.warning("Planner 输出解析失败，降级为单任务计划: error=%s raw=%r", exc, raw[:500])
        plan["tasks"] = [
            _new_task("A", task, [], resolve_tool_hint(tool_hint, available_tools), "react")
        ]
    return plan


def _planner_prompt(available_tools: Collection[str] | None) -> str:
    """Planner 提示词：附上可用工具白名单，约束 tool_hint 只能取真实工具名。"""
    if available_tools:
        listing = "、".join(sorted(set(available_tools)))
        whitelist = f"可用工具白名单（tool_hint 只能取以下之一，取不到就留空字符串）：{listing}"
    else:
        whitelist = "可用工具白名单：无（本模式未绑定任何工具，tool_hint 一律留空）"
    return f"{_PLANNER_PROMPT}\n{whitelist}\n"


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
    except (ValueError, AttributeError, TypeError) as exc:
        # 仅解析类失败才降级（标注失败任务避免无限重规划）；护栏拦截/配额中断必须冒泡
        logger.warning("Replanner 输出解析失败，失败任务标注为 failed: error=%s", exc)
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
            "revisionNo": len(plan["revisions"]) + 1,
            "reason": "，".join(failed_task_ids),
            "changedTaskIds": [t["id"] for t in revised],
        }
    )


# ── 计划干预（plan_edit）───────────────────────────────


def apply_plan_edit(plan: dict[str, Any], plan_edit: dict[str, Any] | None) -> dict[str, Any]:
    """合并用户计划干预（resume 透传），并做干预窗口校验。

    plan_edit 为上行请求体（camelCase）：
    {remove: [taskId], reorder: [taskId...], add: {description, dependsOn, toolHint?, paradigm?}}

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
                [str(d) for d in (add_spec.get("dependsOn") or [])],
                add_spec.get("toolHint"),
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
