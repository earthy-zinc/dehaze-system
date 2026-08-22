"""AI 对话定时任务配置实体（F-M08-009）。

对应表 sys_ai_schedule：将对话中确认的处理流程固化为 Cron 定时任务。
配置类表使用逻辑删除（任务删除后不可恢复，无业务唯一键，标准软删即可）。
"""

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, Index, Integer, SmallInteger, String
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysAiSchedule(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_ai_schedule"
    __table_args__ = (
        Index("idx_user", "user_id"),
        Index("idx_next_trigger", "next_trigger_time"),
        {"comment": "AI对话定时任务配置表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="归属用户ID(关联sys_user.id)"
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, comment="任务名称")
    cron: Mapped[str] = mapped_column(
        String(64), nullable=False, comment='Cron触发规则(5位Cron表达式,如"0 9 * * *")'
    )
    timezone: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="Asia/Shanghai",
        comment="任务时区(触发时间计算时区,默认Asia/Shanghai)",
    )
    input: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="输入来源JSON({type:fixed固定输入|dynamic动态拉取,...})"
    )
    output: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="输出目标JSON(消息推送/回调等)"
    )
    enabled: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="用户启停(1:启用;0:停用)"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        default=1,
        comment="任务状态(1:正常;2:熔断停用,连续失败自动停用)",
    )
    circuit_streak: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="连续失败计数(达到阈值自动熔断停用,重新启用后清零)",
    )
    next_trigger_time: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="下次触发时间(按任务时区计算,供排序与预览)"
    )
