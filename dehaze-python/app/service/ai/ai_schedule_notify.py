"""定时调度执行结果站内信通知（F-M08-009）

无人值守任务每次执行/跳过/失败后向任务归属用户推送站内信，
包含执行结果摘要（成功含消耗与耗时，失败/跳过含原因），
连续失败达到阈值时触发通知升级，提示用户检查任务配置。

对外唯一稳定接口：
    notify_run_result(db, schedule, run)
由 ScheduleExecutor 在每次触发完成后调用；本模块仅承载文案与投递参数。

通知结构对齐 `ai_model_service._notify_model_replacement`：
type=business、priority=3、bizModule="ai_schedule"。
bizId 绑定执行批次（run.id，唯一），保证每次执行都能独立投递通知——
若绑 schedule_id（任务级固定值）会被 MessageService 按 (bizModule, bizId, recipient)
去重，导致同一任务仅首次执行收到通知，后续执行被静默抑制。
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_schedule import SysAiSchedule
from app.models.entity.sys_ai_schedule_run import SysAiScheduleRun
from app.service.message_service import MessageService

# 失败原因展示长度上限（避免超长堆栈刷屏）
ERROR_MSG_MAX_LEN = 200

# 连续失败升级预警阈值（任务可能配置异常，引导检查输入/输出）
CIRCUIT_WARNING_STREAK = 3

# 通知站内信业务模块（用于前端跳转；bizId 用 run.id 保证每次执行独立投递）
_BIND_MODULE = "ai_schedule"

# 跳过原因 -> 通知说明
_SKIP_REASON_MSG = {
    "overlap": "上一次执行尚未完成，本次触发已跳过（任务重叠）。",
    "quota": "积分配额不足，本次执行已跳过，请及时充值或升级会员。",
    "circuit": "任务因连续失败已被熔断停用，本次触发已跳过。",
    "disabled": "任务已停用，本次触发已跳过。",
    "idempotent": "该触发窗口已执行过，本次重复触发已去重跳过。",
}


def _render_success(schedule: SysAiSchedule, run: SysAiScheduleRun) -> tuple[str, str]:
    credits = float(run.credits) if run.credits is not None else 0
    duration_ms = run.duration_ms if run.duration_ms is not None else 0
    jump = (
        f"本次执行产生了会话（ID：{run.conversation_id}），可前往对话记录查看执行结果。"
        if run.conversation_id
        else "本次执行未产生关联会话。"
    )
    return (
        f"定时任务「{schedule.name}」执行成功",
        (
            f"定时任务「{schedule.name}」执行成功，消耗 {credits} 积分，"
            f"耗时 {duration_ms} 毫秒。{jump}"
        ),
    )


def _render_failure(schedule: SysAiSchedule, run: SysAiScheduleRun) -> tuple[str, str]:
    reason = (run.error_msg or "未知错误").strip()[:ERROR_MSG_MAX_LEN]
    return (
        f"定时任务「{schedule.name}」执行失败",
        f"定时任务「{schedule.name}」执行失败，原因：{reason}。",
    )


def _render_circuit_warning(schedule: SysAiSchedule) -> tuple[str, str]:
    """连续失败达阈值（3 次）通知升级预警：提示任务可能配置异常。"""
    return (
        f"定时任务「{schedule.name}」连续失败预警",
        f"定时任务「{schedule.name}」已连续失败 {CIRCUIT_WARNING_STREAK} 次，"
        "任务可能配置异常，请检查输入来源与输出目标配置是否正确。",
    )


def _render_circuit_breaker(schedule: SysAiSchedule) -> tuple[str, str]:
    """连续失败达熔断阈值（5 次）自动停用：引导修复后重新启用。"""
    return (
        f"定时任务「{schedule.name}」已自动停用",
        f"定时任务「{schedule.name}」已连续失败 {schedule.circuit_streak} 次，"
        "系统已自动停用该任务。"
        "请修复输入来源与输出目标配置后，在任务列表中重新启用。",
    )


def _render_skip(run: SysAiScheduleRun) -> tuple[str, str]:
    msg = _SKIP_REASON_MSG.get(run.skip_reason or "", "本次触发已跳过。")
    return ("定时任务触发跳过", msg)


async def notify_run_result(
    db: AsyncSession,
    schedule: SysAiSchedule,
    run: SysAiScheduleRun,
) -> list[int]:
    """向任务归属用户推送执行结果站内信。

    根据执行历史状态/跳过原因渲染对应文案（成功/失败/跳过），
    失败且连续失败计数达熔断阈值（>=5）时升级为"熔断停用"文案，
    达预警阈值（>=3）时升级为"连续失败预警"文案。

    Args:
        db: 数据库会话
        schedule: 定时任务配置
        run: 本次执行/跳过记录

    Returns:
        创建的消息 ID 列表
    """
    if run.status == 1:
        title, content = _render_success(schedule, run)
    elif run.status == 2:
        # 熔断停用：调度执行器在连续失败达阈值时已置 status=2，据此升级为熔断停用文案
        if schedule.status == 2:
            title, content = _render_circuit_breaker(schedule)
        elif schedule.circuit_streak >= CIRCUIT_WARNING_STREAK:
            title, content = _render_circuit_warning(schedule)
        else:
            title, content = _render_failure(schedule, run)
    else:
        title, content = _render_skip(run)

    return await MessageService.send(
        db,
        {
            "type": "business",
            "recipientIds": [schedule.user_id],
            "bizModule": _BIND_MODULE,
            "bizId": str(run.id),
            "priority": 3,
            "title": title,
            "content": content,
            "jumpUrl": f"/scheduled-tasks/{schedule.id}/runs",
        },
    )
