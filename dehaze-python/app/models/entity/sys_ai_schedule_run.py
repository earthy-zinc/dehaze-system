"""AI 对话定时任务执行历史实体（F-M08-009）。

对应表 sys_ai_schedule_run：每次触发/跳过/执行均写入一行。
日志类表，只追加不逻辑删除；保留 30 天由定时任务物理清理。
幂等防重入依赖 uk_schedule_window(schedule_id, window_start) 唯一约束。
"""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, Integer, SmallInteger, String
from sqlalchemy.dialects.mysql import DECIMAL
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import AppendOnlyModel


class SysAiScheduleRun(AppendOnlyModel):
    __tablename__ = "sys_ai_schedule_run"
    __table_args__ = (
        Index("uk_schedule_window", "schedule_id", "window_start", unique=True),
        Index("idx_user", "user_id"),
        {"comment": "AI对话定时任务执行历史表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    schedule_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="关联定时任务ID(关联sys_ai_schedule.id)"
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="归属用户ID(关联sys_user.id,幂等键组成部分)"
    )
    window_start: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, comment="触发窗口(幂等键组成部分,按任务周期对齐)"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=0, comment="执行结果(0:执行中;1:成功;2:失败;3:跳过)"
    )
    skip_reason: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="跳过原因(overlap:任务重叠;quota:配额不足;circuit:熔断停用;idempotent:幂等去重)",
    )
    credits: Mapped[float | None] = mapped_column(
        DECIMAL(10, 4), nullable=True, comment="本次执行消耗积分"
    )
    duration_ms: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="执行耗时(毫秒)"
    )
    error_msg: Mapped[str | None] = mapped_column(String(1000), nullable=True, comment="失败原因")
    conversation_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="执行产生的会话ID(关联sys_ai_conversation.id)"
    )
    request_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="调用链路ID(关联日志排查)"
    )
