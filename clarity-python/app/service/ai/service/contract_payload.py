"""内部结构 → 契约载荷。

计划与中断在系统内部是 snake_case 的执行结构（执行器、中断点直接消费），对外契约是
camelCase 的 Plan/Interrupt。归一集中在本模块，plan.update 事件、pending_interrupt
落库、GET /messages/{id}/interrupt 共用同一出口，避免两条下发路径形状不一致。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any

from app.models.base import _shanghai_now

# 中断可续窗口（天）：与推理检查点 TTL 对齐，超期后 actionable=false
INTERRUPT_TTL_DAYS = 30

# 内部计划状态 → 契约 PlanStatus
_PLAN_STATUS = {
    "pending": "pending",
    "executing": "running",
    "revised": "running",
    "done": "completed",
    "failed": "failed",
}

# 内部任务状态 → 契约 PlanStatus
_TASK_STATUS = {
    "pending": "pending",
    "running": "running",
    "executing": "running",
    "done": "completed",
    "completed": "completed",
    "failed": "failed",
}

_INTERRUPT_TYPES = ("confirm", "quota", "async_wait", "plan_approve")


def plan_status(raw: Any) -> str:
    return _PLAN_STATUS.get(str(raw), "pending")


def task_status(raw: Any) -> str | None:
    if raw is None:
        return None
    return _TASK_STATUS.get(str(raw), "pending")


def _iso(value: datetime) -> str:
    return value.isoformat()


def to_contract_plan(
    plan: dict[str, Any],
    *,
    phase: str | None = None,
    awaiting_approval: bool | None = None,
    actor: str = "assistant",
    now: datetime | None = None,
) -> dict[str, Any]:
    """内部计划 → 契约 Plan（PlanTask/PlanRevision 只保留契约字段）。

    契约 PlanTask 不表达 toolHint/result，这两项纯属执行器内部细节，在此丢弃；
    修订记录补齐 actor 与 at（谁在何时改的计划），缺失时按推理方与当前时间填充。
    时间统一取服务端上海时区（与 create_time 落库同源），避免部署时区不同导致
    契约时间与审计时间错位。
    """
    current = now or _shanghai_now()
    tasks: list[dict[str, Any]] = []
    for task in plan.get("tasks") or []:
        item: dict[str, Any] = {
            "id": str(task.get("id")),
            "description": task.get("description"),
            "dependsOn": list(task.get("depends_on") or []),
            "status": task_status(task.get("status", "pending")),
            "paradigm": task.get("paradigm"),
        }
        tasks.append(item)

    revisions: list[dict[str, Any]] = []
    for revision in plan.get("revisions") or []:
        revisions.append(
            {
                "revisionNo": int(revision.get("revisionNo") or len(revisions) + 1),
                "reason": str(revision.get("reason") or ""),
                "actor": revision.get("actor") or actor,
                "at": revision.get("at") or _iso(current),
            }
        )

    status = plan_status(plan.get("status"))
    return {
        "phase": phase,
        "status": status,
        "tasks": tasks,
        "revisions": revisions,
        "awaitingApproval": (
            awaiting_approval if awaiting_approval is not None else status == "pending"
        ),
    }


def _recommendation(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or not raw.get("modelId") or not raw.get("modelName"):
        return None
    return {
        "modelId": raw["modelId"],
        "modelName": raw["modelName"],
        "reason": raw.get("reason"),
    }


def _alternatives(raw: Any) -> list[dict[str, Any]] | None:
    if not isinstance(raw, list):
        return None
    items = [a for a in (_recommendation(x) for x in raw) if a is not None]
    return items or None


def _interrupt_plan(raw: Any) -> list[dict[str, Any]] | None:
    """plan_approve 中断携带的计划：只取任务项（契约 Interrupt.plan 为 PlanTask[]）。"""
    if isinstance(raw, dict):
        raw = raw.get("tasks")
    if not isinstance(raw, list):
        return None
    tasks = to_contract_plan({"tasks": raw})["tasks"]
    return tasks or None


def build_pending_interrupt(
    record: dict[str, Any] | None,
    *,
    now: datetime | None = None,
    interrupt_id: str | None = None,
) -> dict[str, Any] | None:
    """运行态中断记录 → 契约 Interrupt（持久化与下发共用）。

    record 为 interrupt_handler 保存的形状 {type, stream_session_id, data}；
    id 与 createdAt 在中断发生时生成一次并随消息落库，之后读库即为同一份事实。
    """
    if not record:
        return None
    # 与 create_time/update_time 同一时区基准：全库审计时间是上海时间，契约时间不得用
    # 宿主本地时间（容器常为 UTC），否则 createdAt 与消息 createTime 相差整小时区
    current = now or _shanghai_now()
    inner = record.get("data") if isinstance(record.get("data"), dict) else {}
    # 运行时中断点有两层形状：图事件里的 value 是扁平 {type, data}；Redis 记录由
    # save_interrupt(thread_id, type, value) 包了一层（{type, data: value}）。取到最内层业务载荷。
    if isinstance(inner.get("data"), dict):
        inner = inner["data"]
    itype = str(record.get("type") or inner.get("type") or "confirm")
    if itype not in _INTERRUPT_TYPES:
        itype = "confirm"
    return {
        "id": interrupt_id or str(inner.get("id") or uuid.uuid4().hex),
        "type": itype,
        "confirmKind": inner.get("confirmKind"),
        "action": inner.get("action"),
        "recommendation": _recommendation(inner.get("recommendation")),
        "alternatives": _alternatives(inner.get("alternatives")),
        "upgradeTip": inner.get("upgradeTip"),
        "usedDaily": inner.get("usedDaily"),
        "dailyLimit": inner.get("dailyLimit"),
        "usedMonthly": inner.get("usedMonthly"),
        "monthlyLimit": inner.get("monthlyLimit"),
        "estDuration": inner.get("estDuration"),
        "imageCount": inner.get("imageCount"),
        "plan": _interrupt_plan(inner.get("plan")),
        "reason": inner.get("reason"),
        # impact 是面向用户的处置影响说明，契约无同名字段，落到 detail
        "detail": inner.get("detail") or inner.get("impact"),
        "createdAt": _iso(current),
        "expiresAt": _iso(current + timedelta(days=INTERRUPT_TTL_DAYS)),
        "actionable": True,
    }
