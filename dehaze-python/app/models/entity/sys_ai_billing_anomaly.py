from datetime import datetime

from sqlalchemy import BigInteger, DateTime, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import AppendOnlyModel


class SysAiBillingAnomaly(AppendOnlyModel):
    """AI计费异常事件记录表(只追加，不使用逻辑删除)"""

    __tablename__ = "sys_ai_billing_anomaly"
    __table_args__ = {"comment": "AI计费异常事件记录表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="用户ID(关联sys_user.id)"
    )
    billing_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="计费记录ID(关联sys_ai_billing.id,配额不足类异常无关联记录)"
    )
    anomaly_type: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="异常类型(single_high;burst;consecutive_quota_fail;empty_high_output)"
    )
    detail: Mapped[str] = mapped_column(
        String(255), nullable=False, comment="异常详情"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=0, comment="处理状态(0:待处理;1:已处理;2:已忽略)"
    )
    trigger_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, comment="触发时间"
    )
