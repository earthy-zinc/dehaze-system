from datetime import date, datetime

from sqlalchemy import BigInteger, Date, DateTime, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysReconciliation(BaseModel, SoftDeleteMixin):
    """渠道对账差异记录（每日对账任务产出，运营跟进处理）"""

    __tablename__ = "sys_reconciliation"
    __table_args__ = {"comment": "渠道对账差异表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    recon_date: Mapped[date] = mapped_column(Date, nullable=False, comment="对账日期")
    channel: Mapped[str] = mapped_column(String(16), nullable=False, comment="支付渠道")
    flow_no: Mapped[str] = mapped_column(String(64), nullable=False, comment="支付流水号")
    order_no: Mapped[str | None] = mapped_column(String(32), nullable=True, comment="关联订单号")
    system_amount: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="系统侧金额(分,channel_only 时为空)"
    )
    channel_amount: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="渠道侧金额(分,system_only 时为空)"
    )
    diff_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="差异类型(amount_mismatch:金额不符;system_only:系统多单;channel_only:渠道多单)",
    )
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT,
        nullable=False,
        default=0,
        comment="处理状态(0:未处理;1:已处理)",
    )
    handle_remark: Mapped[str | None] = mapped_column(
        String(256), nullable=True, comment="处理备注"
    )
    handle_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, comment="处理时间")
    handler_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, comment="处理人ID")
