"""定时调度执行引擎（F-M08-009）—— 无人值守执行链路与可靠性

调度基础为 XXL-Job（app/infrastructure/job/handlers.py 的 aiScheduleTrigger 每分钟
扫描触发），本引擎承担单次触发到执行完成的全链路，逐条落实后端实现文档 §2.4 的
可靠性语义：

- 幂等防重入：以 (schedule_id, window_start) 为幂等键写入执行历史，uk_schedule_window
  唯一约束兜底，多实例并发扫描/时钟漂移/重启均不重复执行（不依赖内存状态）。
- 并发控制：单任务 Redis 运行标记（ai:schedule:{id}:running）防任务重叠；模块级
  asyncio.Semaphore 限制平台级全局并发，防止任务风暴打爆推理服务。
- 失败分级重试：临时错误（网络/超时/限流）指数退避重试 3 次（1s/2s/4s）；
  不可重试错误（参数/权限/配置类 BusinessException）直接失败。
- 连续失败熔断：同一任务连续失败 >= 5 次自动停用（status=2）并通知；成功清零；
  熔断停用/用户停用期间触发跳过（skip_reason=circuit/disabled）。
- 配额保护：执行前校验用户日/月限额，不足跳过（skip_reason=quota）并通知。
- 执行观测：每次触发/跳过/执行均写执行历史；执行完成推进 next_trigger_time。

无人值守执行本体复用内部推理链路：创建任务专属会话 -> 组装输入 -> 调用
AiMessageService.send_message 发起推理（其内部创建 user/assistant 消息并起后台
reasoning 任务）-> 消费返回的 StreamingResponse.body_iterator 驱动推理至完成
（事件仅落 Redis 缓存，无 HTTP/SSE 网络传输）-> 读取 assistant 消息的
credits/conversation_id/status/error/content 写入执行历史。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_schedule import SysAiSchedule
from app.models.entity.sys_ai_schedule_run import SysAiScheduleRun
from app.models.schema.ai_conversation import ConversationCreate, MessageSend
from app.repository.ai_message_repository import ai_message_repository
from app.repository.ai_schedule_repository import ai_schedule_repository
from app.repository.ai_schedule_run_repository import ai_schedule_run_repository
from app.service.ai.ai_schedule_notify import notify_run_result
from app.service.ai_conversation_service import AiConversationService
from app.service.ai_message_service import AiMessageService
from app.service.billing.quota_service import quota_service

logger = logging.getLogger(__name__)

# ==================== 配置（收编 app/config.py Settings）====================

# 平台级全局并发上限（对齐 §2.4 ai.schedule.max-concurrent）
MAX_CONCURRENT = settings.AI_SCHEDULE_MAX_CONCURRENT
# 单任务运行标记 TTL（秒），防进程崩溃后标记泄漏；亦作执行中批次僵尸回收阈值
RUN_MARK_TTL = settings.AI_SCHEDULE_RUN_MARK_TTL
# 连续失败熔断阈值（>= 该值自动停用）
CIRCUIT_THRESHOLD = settings.AI_SCHEDULE_CIRCUIT_THRESHOLD
# 临时错误指数退避重试：次数与退避间隔（秒）
RETRY_MAX = settings.AI_SCHEDULE_RETRY_MAX
RETRY_BACKOFF = (1, 2, 4)
# 每次触发配额预检的最小预估积分（limits=0 视为无限额）
QUOTA_ESTIMATE_CREDITS = 1
# 扫描单批任务上限
SCAN_LIMIT = 100

# 运行标记 Redis key 前缀
_RUN_MARK_PREFIX = "ai:schedule:{}:running"

# 临时/可重试异常类型（网络瞬断、连接/读写超时、服务端 5xx 语义）
_RETRYABLE_EXC = (
    asyncio.TimeoutError,
    TimeoutError,
    ConnectionError,
    OSError,
)


class _RetryableError(Exception):
    """临时错误包装（可重试：网络/超时/限流类）。"""


class ScheduleExecutor:
    """定时任务执行引擎（单例）。"""

    def __init__(self) -> None:
        # 平台级全局并发信号量（asyncio.Semaphore）
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # ==================== 扫描触发（XXL-Job aiScheduleTrigger） ====================

    async def scan_and_trigger(self, db: AsyncSession, redis: Redis) -> dict[str, int]:
        """扫描到期任务并逐条触发。

        由 aiScheduleTrigger（每分钟）调用。先回收僵尸批次（执行中超时未完成的
        崩溃残留），再扫描到期任务；get_due_tasks 已过滤启用+正常状态任务；
        每任务使用独立 DB 会话执行，避免幂等插入冲突后的回滚相互污染。
        """
        from app.database import get_db_session

        recovered = await self._recover_stale_runs(db)
        if recovered:
            logger.info("回收僵尸执行批次 %d 条", recovered)
        due = await ai_schedule_repository.get_due_tasks(db, datetime.now(), SCAN_LIMIT)
        summary = {
            "scanned": len(due),
            "triggered": 0,
            "skipped": 0,
            "failed": 0,
            "recovered": recovered,
        }

        for task in due:
            async with get_db_session() as task_db:
                try:
                    result = await self.trigger_once(
                        task_db, redis, task.id, task.user_id, manual=False
                    )
                    if result.get("skipped"):
                        summary["skipped"] += 1
                    else:
                        summary["triggered"] += 1
                except Exception as exc:  # noqa: BLE001 单任务失败不影响整体扫描
                    summary["failed"] += 1
                    logger.warning("定时任务触发失败 schedule_id=%s: %s", task.id, exc)

        return summary

    async def _recover_stale_runs(self, db: AsyncSession) -> int:
        """回收僵尸执行批次。

        执行中(status=0)超过 RUN_MARK_TTL 仍无终态的批次，判定为进程崩溃/强杀残留：
        置为失败（error_msg 注明系统回收）、计入连续失败计数（达阈值熔断）并通知用户。
        返回回收条数。
        """
        threshold = datetime.now() - timedelta(seconds=RUN_MARK_TTL)
        stale = await ai_schedule_run_repository.get_stale_running(db, threshold)
        for run in stale:
            run.status = 2
            run.error_msg = "执行中断（进程异常终止），系统自动回收"
            schedule = await ai_schedule_repository.get_by_id(db, run.schedule_id)
            if schedule is not None:
                schedule.circuit_streak = (schedule.circuit_streak or 0) + 1
                if schedule.circuit_streak >= CIRCUIT_THRESHOLD:
                    await ai_schedule_repository.mark_circuit(db, schedule.id)
        await db.commit()
        for run in stale:
            schedule = await ai_schedule_repository.get_by_id(db, run.schedule_id)
            if schedule is not None:
                try:
                    await notify_run_result(db, schedule, run)
                except Exception:  # noqa: BLE001 回收通知失败不阻断
                    logger.warning("僵尸回收通知失败 run_id=%s", run.id)
        return len(stale)

    # ==================== 单次触发（自动/手动统一入口） ====================

    async def trigger_once(
        self,
        db: AsyncSession,
        redis: Redis,
        schedule_id: int,
        user_id: int,
        *,
        manual: bool = False,
    ) -> dict[str, Any]:
        """触发一次任务执行（自动扫描与手动触发共用同一链路）。

        Args:
            db: 数据库会话
            redis: Redis 客户端
            schedule_id: 任务 ID
            user_id: 触发用户（须为任务归属用户）
            manual: 是否手动触发（不推进 next_trigger_time）

        Returns:
            执行结果摘要 dict
        """
        schedule = await ai_schedule_repository.get_by_id(db, schedule_id)
        if not schedule or schedule.user_id != user_id:
            return self._summary(skipped=True, skip_reason="disabled", msg="任务不存在或无权触发")
        if schedule.enabled != 1 or schedule.status == 2:
            reason = "circuit" if schedule.status == 2 else "disabled"
            return await self._record_skip(db, schedule, reason, manual)

        run = await self._begin_window(db, schedule)
        if run is None:
            # 同窗口已有记录：幂等防重入，跳过不重复执行。
            # 仍推进 next_trigger_time（非手动）：服务重启/时钟漂移追赶期
            # next_trigger_time 停留过去会导致每分钟空转扫描，此处推进跳出追赶循环
            if not manual:
                await self._advance_next_trigger(db, schedule)
                await db.commit()
            return self._summary(skipped=True, skip_reason="idempotent", msg="同窗口已执行过")

        try:
            # 单任务重叠防重入（Redis SET NX EX 运行标记）
            if not await self._acquire_run_mark(redis, schedule.id):
                # 本窗口批次已由 _begin_window 预留，直接标记为 overlap 跳过
                return await self._finalize_skip(db, schedule, run, "overlap", manual)

            # 平台级全局并发限流
            async with self._semaphore:
                return await self._run_with_guards(db, redis, schedule, run, manual)
        finally:
            await self._release_run_mark(redis, schedule.id)

    # ==================== 执行主流程（配额校验/重试/熔断/推进） ====================

    async def _run_with_guards(
        self,
        db: AsyncSession,
        redis: Redis,
        schedule: SysAiSchedule,
        run: SysAiScheduleRun,
        manual: bool,
    ) -> dict[str, Any]:
        # 配额保护：执行前校验日/月限额，不足则跳过并通知
        if not await self._quota_ok(db, schedule.user_id):
            return await self._finalize_skip(db, schedule, run, "quota", manual)

        started = time.monotonic()
        success = False
        last_error = ""
        conversation_id: int | None = None
        credits: float | None = None
        try:
            conversation_id, credits = await self._execute_inference(db, schedule)
            success = True
        except Exception as exc:  # noqa: BLE001 推理失败统一按失败处理
            last_error = str(exc) or type(exc).__name__
            logger.warning("定时任务执行失败 schedule_id=%s: %s", schedule.id, last_error)

        run.status = 1 if success else 2
        run.conversation_id = conversation_id
        run.credits = credits
        run.error_msg = last_error[:1000] if last_error else None
        run.duration_ms = int((time.monotonic() - started) * 1000)

        # 熔断计数：成功清零，失败累加（>= 阈值自动停用）
        if success:
            schedule.circuit_streak = 0
        else:
            schedule.circuit_streak += 1
            if schedule.circuit_streak >= CIRCUIT_THRESHOLD:
                await ai_schedule_repository.mark_circuit(db, schedule.id)
                schedule.status = 2
        await db.commit()

        # 推进下次触发时间（仅自动触发；手动不改变 cron 规则）
        if not manual:
            await self._advance_next_trigger(db, schedule)
            await db.commit()

        # 执行结果通知（成功含消耗与耗时，失败含原因）
        await notify_run_result(db, schedule, run)
        return self._summary(
            skipped=False,
            ok=success,
            run_id=run.id,
            credits=float(run.credits) if run.credits is not None else None,
            duration_ms=run.duration_ms,
            error=last_error or None,
            conversation_id=run.conversation_id,
            circuit_streak=schedule.circuit_streak,
            circuited=not success and schedule.circuit_streak >= CIRCUIT_THRESHOLD,
        )

    # ==================== 幂等窗口与运行标记 ====================

    async def _begin_window(
        self,
        db: AsyncSession,
        schedule: SysAiSchedule,
    ) -> SysAiScheduleRun | None:
        """以幂等键写入执行批次。

        成功插入返回新批次（status=0 执行中，完成时置 1/2/3）；同窗口已存在则
        返回 None（幂等跳过）。status=0 保证进程崩溃残留的批次不会被观测为"成功"，
        由僵尸回收（_recover_stale_runs）判定为失败。
        """
        window_start = self._align_window_start(schedule)
        candidate = SysAiScheduleRun(
            schedule_id=schedule.id,
            user_id=schedule.user_id,
            window_start=window_start,
            status=0,
        )
        inserted = await ai_schedule_run_repository.create_with_window(db, candidate)
        if inserted is not candidate:
            # 唯一约束冲突：返回的是已存在记录，本次触发应去重跳过
            await db.rollback()
            return None
        await db.commit()
        return candidate

    def _align_window_start(self, schedule: SysAiSchedule) -> datetime:
        """计算触发窗口起点（幂等键组成部分）。

        对齐到 cron 当前周期起点：以 now 为基准，取最近一次不晚于 now 的计划触发
        时刻（croniter.get_prev）。同一周期内的自动/手动/重复触发映射到同一窗口，
        保证幂等与"同窗口不重复执行"（T-SC-057 / T-SC-071）。
        """
        tz = self._resolve_tz(schedule.timezone)
        now = datetime.now(tz)
        it = croniter(schedule.cron, now + timedelta(milliseconds=1), ret_type=datetime)
        prev = it.get_prev()
        return prev.replace(tzinfo=None)

    @staticmethod
    def _resolve_tz(timezone: str) -> ZoneInfo:
        try:
            return ZoneInfo(timezone)
        except ZoneInfoNotFoundError:
            return ZoneInfo("Asia/Shanghai")

    async def _acquire_run_mark(self, redis: Redis, schedule_id: int) -> bool:
        return bool(
            await redis.set(_RUN_MARK_PREFIX.format(schedule_id), "1", nx=True, ex=RUN_MARK_TTL)
        )

    async def _release_run_mark(self, redis: Redis, schedule_id: int) -> None:
        await redis.delete(_RUN_MARK_PREFIX.format(schedule_id))

    # ==================== 配额校验 ====================

    async def _quota_ok(self, db: AsyncSession, user_id: int) -> bool:
        daily_limit, monthly_limit = await quota_service.get_limits(db, user_id)
        # limits 均为 0 视为无限额（对齐 get_limits 语义），跳过配额预检
        if daily_limit <= 0 and monthly_limit <= 0:
            return True
        daily_used, monthly_used = await quota_service.get_used(user_id)
        return (
            daily_used + QUOTA_ESTIMATE_CREDITS <= daily_limit
            and monthly_used + QUOTA_ESTIMATE_CREDITS <= monthly_limit
        )

    # ==================== 无人值守推理执行 ====================

    async def _execute_inference(
        self, db: AsyncSession, schedule: SysAiSchedule
    ) -> tuple[int, float | None]:
        """以任务归属用户身份发起推理，复用内部消息链路。

        带失败分级重试：临时错误（网络/超时/限流类）按指数退避重试 RETRY_MAX 次；
        不可重试错误（参数/权限/配置类 BusinessException）不重试。

        Returns:
            (执行产生的会话 ID, 本次执行消耗积分)
        """
        content = self._build_input_text(schedule)
        conversation_id = await self._ensure_conversation(db, schedule, content)

        last_exc: Exception | None = None
        for attempt in range(RETRY_MAX + 1):
            try:
                return await self._send_and_wait(db, conversation_id, schedule.user_id, content)
            except BusinessException as exc:
                # LLM 上游临时故障（AI_LLM_CALL_FAILED：llm_client 已穷尽 Key 轮换
                # 与路由降级仍失败，多为瞬时故障）→ 指数退避重试；
                # 其余业务异常（参数/权限/配置类）不可重试，直接失败
                if exc.code == ResultCode.AI_LLM_CALL_FAILED:
                    retryable = _RetryableError(str(exc))
                else:
                    raise
                last_exc = retryable
                if attempt < RETRY_MAX:
                    await asyncio.sleep(RETRY_BACKOFF[attempt])
                    continue
                raise last_exc from exc
            except Exception as exc:
                # 仅临时错误（网络/超时/限流）进入退避重试
                if not isinstance(exc, _RetryableError):
                    raise
                last_exc = exc
                if attempt < RETRY_MAX:
                    await asyncio.sleep(RETRY_BACKOFF[attempt])
                    continue
                raise last_exc from exc
        raise last_exc or BusinessException("定时任务推理失败")

    async def _ensure_conversation(
        self,
        db: AsyncSession,
        schedule: SysAiSchedule,
        content: str,
    ) -> int:
        """创建任务专属会话（标题含任务名，便于结果跳转）。"""
        conv = await AiConversationService.create_conversation(
            db,
            schedule.user_id,
            ConversationCreate(
                title=f"{schedule.name}-定时执行",
                model=schedule.output.get("model") if schedule.output else None,
                systemPrompt=self._extract_system_prompt(schedule),
            ),
        )
        return conv.id

    async def _send_and_wait(
        self,
        db: AsyncSession,
        conversation_id: int,
        user_id: int,
        content: str,
    ) -> tuple[int, float | None]:
        """发起推理并等待完成。

        调用 AiMessageService.send_message 得到 StreamingResponse，消费其
        body_iterator 驱动内部 reasoning 后台任务至完成（不发起 HTTP/SSE 网络传输，
        事件仅落 Redis 缓存）。推理完成后校验 assistant 消息状态并取消耗积分。

        Returns:
            (会话 ID, 本次执行消耗积分)
        """
        try:
            response = await AiMessageService.send_message(
                db,
                conversation_id,
                user_id,
                MessageSend(content=content, model=None),
                f"sched-{conversation_id}-{uuid.uuid4().hex}",
            )
            # 消费 SSE body_iterator：驱动推理执行至 message.end
            async for _ in response.body_iterator:  # noqa: B007 仅驱动完成
                pass
        except BusinessException:
            raise
        except Exception as exc:  # noqa: BLE001 网络/超时/连接类临时错误
            if isinstance(exc, _RETRYABLE_EXC):
                raise _RetryableError(str(exc)) from exc
            raise

        # 推理结束后读取最近一条 assistant 消息，校验完成状态并取消耗积分
        messages, _ = await ai_message_repository.list_by_conversation(db, conversation_id, 1, 20)
        assistant = next((m for m in reversed(messages) if m.role == "assistant"), None)
        if assistant is None:
            raise BusinessException("推理未生成回复消息")
        if assistant.status == 3:
            raise BusinessException(assistant.error or "推理失败")
        credits = float(assistant.credits) if assistant.credits is not None else None
        return conversation_id, credits

    # ==================== 输入组装 ====================

    def _build_input_text(self, schedule: SysAiSchedule) -> str:
        """根据输入来源组装推理消息内容。

        - fixed：使用预设内容/图片集（input.content 或 input.images 拼 markdown）。
        - dynamic：本期以提示文本降级注入；MCP 动态拉取标注为待扩展。
        """
        cfg = schedule.input or {}
        if cfg.get("type") == "fixed":
            parts: list[str] = []
            if cfg.get("content"):
                parts.append(str(cfg["content"]))
            for img in cfg.get("images") or []:
                parts.append(f"![图片]({img})")
            if parts:
                return "\n".join(parts)
            return f"请执行定时任务「{schedule.name}」的固定输入处理。"
        return (
            f"请执行定时任务「{schedule.name}」："
            f"{cfg.get('prompt', '按任务配置拉取并处理最新数据。')}"
        )

    @staticmethod
    def _extract_system_prompt(schedule: SysAiSchedule) -> str | None:
        """提取会话系统提示词（input.system_prompt），无则用默认。"""
        return (schedule.input or {}).get("system_prompt")

    # ==================== 跳过与收尾 ====================

    async def _record_skip(
        self,
        db: AsyncSession,
        schedule: SysAiSchedule,
        reason: str,
        manual: bool,
    ) -> dict[str, Any]:
        """记录一次跳过（circuit/disabled/overlap）到执行历史并通知。"""
        window_start = self._align_window_start(schedule)
        run = SysAiScheduleRun(
            schedule_id=schedule.id,
            user_id=schedule.user_id,
            window_start=window_start,
            status=3,
            skip_reason=reason,
        )
        inserted = await ai_schedule_run_repository.create_with_window(db, run)
        if inserted is not run:
            await db.rollback()
            # 幂等跳过同样推进 next_trigger_time，避免追赶期每分钟空转扫描
            if not manual:
                await self._advance_next_trigger(db, schedule)
                await db.commit()
            return self._summary(skipped=True, skip_reason="idempotent", msg="同窗口已执行过")
        await db.commit()
        if not manual:
            await self._advance_next_trigger(db, schedule)
            await db.commit()
        await notify_run_result(db, schedule, run)
        return self._summary(skipped=True, skip_reason=reason, msg=_SKIP_HINT.get(reason, "跳过"))

    async def _finalize_skip(
        self,
        db: AsyncSession,
        schedule: SysAiSchedule,
        run: SysAiScheduleRun,
        reason: str,
        manual: bool,
    ) -> dict[str, Any]:
        """将已写入的执行批次标记为跳过（quota），推进触发并通知。"""
        run.status = 3
        run.skip_reason = reason
        await db.commit()
        if not manual:
            await self._advance_next_trigger(db, schedule)
            await db.commit()
        await notify_run_result(db, schedule, run)
        return self._summary(skipped=True, skip_reason=reason, msg=_SKIP_HINT.get(reason, "跳过"))

    async def _advance_next_trigger(self, db: AsyncSession, schedule: SysAiSchedule) -> None:
        """推进下次触发时间（croniter 下次，按任务时区）。"""
        tz = self._resolve_tz(schedule.timezone)
        base = schedule.next_trigger_time or datetime.now()
        if base.tzinfo is None:
            base = base.replace(tzinfo=tz)
        it = croniter(schedule.cron, base, ret_type=datetime)
        nxt = it.get_next()
        await ai_schedule_repository.update_next_trigger(db, schedule.id, nxt.replace(tzinfo=None))

    # ==================== 摘要 ====================

    @staticmethod
    def _summary(**kwargs: Any) -> dict[str, Any]:
        return kwargs


_SKIP_HINT = {
    "overlap": "任务正在执行中",
    "quota": "积分配额不足",
    "circuit": "任务熔断停用",
    "disabled": "任务已停用",
}

schedule_executor = ScheduleExecutor()
