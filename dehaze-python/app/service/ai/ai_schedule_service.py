"""定时调度服务（F-M08-009）

支持用户将对话中确认的处理流程固化为 Cron 定时任务，含任务 CRUD、启停、
软删、到期扫描触发数据的供给，以及 Cron 解释与下次执行时间预览。
"""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import CroniterBadCronError, croniter
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_schedule import SysAiSchedule
from app.models.entity.sys_ai_schedule_run import SysAiScheduleRun
from app.models.schema.ai_schedule import (
    NextTimesPreview,
    RunHistoryItem,
    RunSummary,
    ScheduleCreate,
    ScheduleDetail,
    ScheduleListItem,
    SchedulePageQuery,
    ScheduleUpdate,
)
from app.models.schema.common import PageResult
from app.repository.ai_schedule_repository import ai_schedule_repository
from app.repository.ai_schedule_run_repository import ai_schedule_run_repository
from app.repository.base import escape_like
from app.repository.member_repository import member_repository

# 会员等级编码 -> 数值等级（定时调度仅 VIP2+ 可用，无会员记录视为 level_0）
_LEVEL_MAP = {"level_0": 0, "level_1": 1, "level_2": 2, "level_3": 3}

# 单用户定时任务上限
MAX_SCHEDULES_PER_USER = 20

# 默认任务时区（与配额重置时区一致）
DEFAULT_TIMEZONE = "Asia/Shanghai"


class ScheduledTaskService:
    async def create(self, db: AsyncSession, user_id: int, form: ScheduleCreate) -> ScheduleDetail:
        """创建定时任务：校验 VIP2+、单用户上限 20、Cron 合法性，并计算下次触发时间。"""
        await self._ensure_vip2(db, user_id)

        count = await ai_schedule_repository.count_by_user(db, user_id)
        if count >= MAX_SCHEDULES_PER_USER:
            raise BusinessException(
                ResultCode.DATA_STATE_NOT_ALLOW, f"定时任务数量已达上限({MAX_SCHEDULES_PER_USER}个)"
            )

        timezone = form.timezone or DEFAULT_TIMEZONE
        cron = self._normalize_cron(form.cron)
        self._parse_cron(cron)
        next_trigger = self._compute_next_trigger(cron, timezone)

        task = SysAiSchedule(
            user_id=user_id,
            name=form.name.strip(),
            cron=cron,
            timezone=timezone,
            input=form.input,
            output=form.output,
            enabled=1,
            status=1,
            circuit_streak=0,
            next_trigger_time=next_trigger,
        )
        await ai_schedule_repository.create(db, task)
        return self._to_detail(task)

    async def update(
        self, db: AsyncSession, user_id: int, schedule_id: int, form: ScheduleUpdate
    ) -> ScheduleDetail:
        """更新任务：可更新名称/Cron/时区/输入输出/启停，变更后重算下次触发时间。"""
        task = await self._get_owned(db, user_id, schedule_id)

        if form.name is not None:
            task.name = form.name.strip()
        if form.cron is not None:
            cron = self._normalize_cron(form.cron)
            self._parse_cron(cron)
            task.cron = cron
        if form.timezone is not None:
            task.timezone = form.timezone
        if form.input is not None:
            task.input = form.input
        if form.output is not None:
            task.output = form.output
        if form.enabled is not None:
            task.enabled = form.enabled

        if form.cron is not None or form.timezone is not None or form.enabled == 1:
            task.next_trigger_time = self._compute_next_trigger(
                task.cron, task.timezone
            )

        await db.flush()
        await db.refresh(task)
        return self._to_detail(task)

    async def get_detail(self, db: AsyncSession, user_id: int, schedule_id: int) -> ScheduleDetail:
        task = await self._get_owned(db, user_id, schedule_id)
        return self._to_detail(task)

    async def list_page(
        self,
        db: AsyncSession,
        user_id: int,
        query: SchedulePageQuery,
    ) -> PageResult[ScheduleListItem]:
        """任务列表（按下次触发时间排序，批量聚合最近执行摘要，避免 N+1）。"""
        stmt = select(SysAiSchedule).where(SysAiSchedule.user_id == user_id)
        if query.keyword:
            stmt = stmt.where(
                SysAiSchedule.name.like(f"%{escape_like(query.keyword)}%", escape="\\")
            )
        # MySQL 不支持 NULLS LAST，用 `ISNULL(col)` 做 NULL 排后的等价排序
        stmt = stmt.order_by(
            SysAiSchedule.enabled.desc(),
            func.isnull(SysAiSchedule.next_trigger_time).asc(),
            SysAiSchedule.next_trigger_time.asc(),
            SysAiSchedule.id.asc(),
        )
        items, total = await ai_schedule_repository.paginate(
            db, stmt, query.pageNum, query.pageSize
        )

        latest_map = await ai_schedule_run_repository.get_latest_by_schedule_ids(
            db, [t.id for t in items]
        )

        rows = [
            ScheduleListItem(
                **self._to_detail(t).model_dump(by_alias=True),
                lastRun=self._to_run_summary(latest_map.get(t.id)),
            )
            for t in items
        ]
        return PageResult[ScheduleListItem](list=rows, total=total)

    async def set_enabled(self, db: AsyncSession, user_id: int, schedule_id: int, enabled: int) -> None:
        """启停任务。

        - 启用(true)：若当前熔断停用(status=2)则同时 reset_circuit（清零计数+status=1）；
          并重算下次触发时间。
        - 停用(false)：仅置 enabled=0，保留熔断状态与下次触发时间。
        """
        task = await self._get_owned(db, user_id, schedule_id)
        if enabled == 1:
            if task.status == 2:
                await ai_schedule_repository.reset_circuit(db, schedule_id)
            await ai_schedule_repository.update_next_trigger(
                db,
                schedule_id,
                self._compute_next_trigger(task.cron, task.timezone),
            )
        await ai_schedule_repository.set_enabled(db, schedule_id, enabled)

    async def delete(self, db: AsyncSession, user_id: int, schedule_id: int) -> None:
        """删除任务（软删除，删除后不可恢复）。"""
        await self._get_owned(db, user_id, schedule_id)
        await ai_schedule_repository.soft_delete(db, schedule_id)

    async def preview_next_times(
        self, cron: str, count: int = 5, timezone: str = DEFAULT_TIMEZONE
    ) -> NextTimesPreview:
        """Cron 解释与接下来 N 次触发时间预览。非法 Cron 抛参数异常。"""
        cron = self._normalize_cron(cron)
        self._parse_cron(cron)
        description = self._describe_cron(cron)
        next_times = self._compute_next_times(cron, timezone, count)
        return NextTimesPreview(description=description, nextTimes=next_times)

    async def list_history(
        self,
        db: AsyncSession,
        user_id: int,
        schedule_id: int,
        page: int,
        size: int,
    ) -> PageResult[RunHistoryItem]:
        """执行历史分页（先做归属校验，再按时间倒序分页）。"""
        await self._get_owned(db, user_id, schedule_id)
        items, total = await ai_schedule_run_repository.page_by_schedule(
            db, schedule_id, page, size
        )
        rows = [self._to_run_history(run) for run in items]
        return PageResult[RunHistoryItem](list=rows, total=total)

    # ── 内部工具 ──────────────────────────────────────────

    async def _ensure_vip2(self, db: AsyncSession, user_id: int) -> None:
        """校验用户为 VIP2 及以上（无会员记录视为 level_0，不允许创建定时任务）。"""
        member = await member_repository.get_by_user_id(db, user_id)
        level = _LEVEL_MAP.get(member.level_code, 0) if member else 0
        if level < 2:
            raise BusinessException(
                ResultCode.OPERATION_NOT_ALLOW, "定时调度功能需 VIP2 及以上会员，请升级会员后使用"
            )

    async def _get_owned(self, db: AsyncSession, user_id: int, schedule_id: int) -> SysAiSchedule:
        """取归属用户且未删除的任务，越权或不存在抛异常。"""
        task = await ai_schedule_repository.get_by_id(db, schedule_id)
        if not task or task.user_id != user_id:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "定时任务不存在")
        return task

    def _parse_cron(self, cron: str) -> None:
        """校验 5 位 Cron 表达式合法性（分钟 小时 日 月 星期）。"""
        try:
            croniter(cron)
        except (CroniterBadCronError, ValueError, KeyError) as exc:
            raise BusinessException(ResultCode.PARAM_ERROR, f"Cron 表达式非法: {cron}") from exc

    # 常用频率标识的星期别名（对齐 cron 语义：0 与 7 均为周日，1=周一 … 6=周六）
    _WEEKDAY_ALIAS = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 0}

    def _normalize_cron(self, raw: str) -> str:
        """归一化触发规则为标准 5 位 Cron 表达式。

        支持两种输入（对齐 API 契约 §2.8.4：Cron 表达式或常用频率标识）：
        - 标准 5 位 Cron 表达式（含 @hourly/@daily 等别名按原样透传给 croniter）
        - 常用频率标识（存储前统一转换为标准 Cron，扫描器/预览无需感知）：
          * ``daily@HH:MM``          每天 HH:MM          → ``M H * * *``
          * ``weekly@D@HH:MM``       每周 D 的 HH:MM     → ``M H * * D``
            （D 支持 mon/tue/wed/thu/fri/sat/sun 或 0-6，按 cron 语义：0 与 7 为周日）
          * ``monthly@D@HH:MM``      每月 D 号的 HH:MM   → ``M H D * *``

        无法识别的格式原样返回，由 _parse_cron 统一抛参数异常。
        """
        text = raw.strip()
        if "@" not in text:
            return text
        parts = text.split("@")
        if len(parts) not in (2, 3) or parts[0] not in ("daily", "weekly", "monthly"):
            return text

        def _parse_hm(hm: str) -> tuple[int, int]:
            try:
                hh, mm = hm.split(":")
                hour, minute = int(hh), int(mm)
            except ValueError as exc:
                raise BusinessException(
                    ResultCode.PARAM_ERROR, f"触发规则时间格式非法: {raw}"
                ) from exc
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise BusinessException(ResultCode.PARAM_ERROR, f"触发规则时间超出范围: {raw}")
            return hour, minute

        hour, minute = _parse_hm(parts[-1])
        if parts[0] == "daily":
            return f"{minute} {hour} * * *"
        day = parts[1].strip().lower()
        if parts[0] == "weekly":
            weekday = self._WEEKDAY_ALIAS.get(day)
            if weekday is None:
                try:
                    weekday = int(day) % 7
                except ValueError:
                    return raw.strip()
            return f"{minute} {hour} * * {weekday}"
        # monthly：1-31 号
        try:
            dom = int(day)
        except ValueError:
            return text
        if not (1 <= dom <= 31):
            raise BusinessException(ResultCode.PARAM_ERROR, f"触发规则日期超出范围: {raw}")
        return f"{minute} {hour} {dom} * *"

    def _compute_next_trigger(self, cron: str, timezone: str) -> datetime:
        """计算下一次触发时间（按任务时区，返回该时区的本地时间用于落库排序）。"""
        tz = self._resolve_tz(timezone)
        it = croniter(cron, datetime.now(tz), ret_type=datetime)
        return it.get_next().replace(tzinfo=None)

    def _compute_next_times(self, cron: str, timezone: str, count: int) -> list[datetime]:
        """返回接下来 N 次触发时间（带时区的 ISO 时间）。"""
        tz = self._resolve_tz(timezone)
        it = croniter(cron, datetime.now(tz), ret_type=datetime)
        return [it.get_next() for _ in range(count)]

    def _resolve_tz(self, timezone: str) -> ZoneInfo:
        try:
            return ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise BusinessException(ResultCode.PARAM_ERROR, f"任务时区非法: {timezone}") from exc

    def _describe_cron(self, cron: str) -> str:
        """Cron 的人类可读描述（自研简版映射）。

        覆盖：每小时/每天/每周几/每月几号/每月的H点M分；无法归类时返回原始表达式。
        """
        try:
            parts = cron.split()
            if len(parts) != 5:
                return cron
            minute, hour, day, month, weekday = parts
            weekday_num = {
                "0": "日",
                "1": "一",
                "2": "二",
                "3": "三",
                "4": "四",
                "5": "五",
                "6": "六",
                "7": "日",
            }

            # 每分钟
            if minute == "*" and hour == "*" and day == "*" and month == "*" and weekday == "*":
                return "每分钟"
            # 每小时（整点或指定分钟）
            if hour == "*" and day == "*" and month == "*" and weekday == "*":
                if minute == "0":
                    return "每小时整点"
                return f"每小时 {self._fmt_minute(minute)}分"
            # 每天 H 点 M 分
            if hour != "*" and day == "*" and month == "*" and weekday == "*":
                return (
                    f"每天 {self._fmt_hour(hour)}点"
                    f"{self._fmt_minute(minute)}分"
                )
            # 每周几
            if day == "*" and month == "*" and weekday != "*":
                days = "、".join(
                    f"周{weekday_num[w]}" for w in weekday.split(",") if w in weekday_num
                )
                time_part = (
                    f"{self._fmt_hour(hour)}点"
                    f"{self._fmt_minute(minute)}分"
                    if hour != "*"
                    else f"每小时{self._fmt_minute(minute)}分"
                )
                return f"每周{days} {time_part}"
            # 每月几号
            if month == "*" and weekday == "*" and day != "*":
                days = "、".join(f"{d}号" for d in day.split(","))
                time_part = (
                    f"{self._fmt_hour(hour)}点"
                    f"{self._fmt_minute(minute)}分"
                    if hour != "*"
                    else f"每小时{self._fmt_minute(minute)}分"
                )
                return f"每月{days} {time_part}"
            # 每月的第几天（指定月份）
            if month != "*" and weekday == "*" and day != "*":
                months = "、".join(f"{m}月" for m in month.split(","))
                days = "、".join(f"{d}号" for d in day.split(","))
                return (
                    f"每年{months}{days} {self._fmt_hour(hour)}点"
                    f"{self._fmt_minute(minute)}分"
                )
        except Exception:
            pass
        return f"Cron({cron})"

    def _fmt_hour(self, hour: str) -> str:
        return hour.zfill(2)

    def _fmt_minute(self, minute: str) -> str:
        return "00" if minute == "*" else minute.zfill(2)

    def _to_detail(self, task: SysAiSchedule) -> ScheduleDetail:
        return ScheduleDetail(
            id=task.id,
            userId=task.user_id,
            name=task.name,
            cron=task.cron,
            timezone=task.timezone,
            input=task.input,
            output=task.output,
            enabled=task.enabled,
            status=task.status,
            circuitStreak=task.circuit_streak,
            nextTriggerTime=task.next_trigger_time,
            createTime=task.create_time,
        )

    def _to_run_summary(self, run: SysAiScheduleRun | None) -> RunSummary | None:
        if run is None:
            return None
        return RunSummary(
            status=run.status,
            skipReason=run.skip_reason,
            credits=float(run.credits) if run.credits is not None else None,
            durationMs=run.duration_ms,
            errorMsg=run.error_msg,
            conversationId=run.conversation_id,
            createTime=run.create_time,
        )

    def _to_run_history(self, run: SysAiScheduleRun) -> RunHistoryItem:
        return RunHistoryItem(
            id=run.id,
            scheduleId=run.schedule_id,
            status=run.status,
            skipReason=run.skip_reason,
            credits=float(run.credits) if run.credits is not None else None,
            durationMs=run.duration_ms,
            errorMsg=run.error_msg,
            conversationId=run.conversation_id,
            requestId=run.request_id,
            windowStart=run.window_start,
            createTime=run.create_time,
        )


# 定时调度服务单例
scheduled_task_service = ScheduledTaskService()
