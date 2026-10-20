from datetime import datetime

from sqlalchemy import BigInteger, DateTime, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysBalanceRefund(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_balance_refund"
    __table_args__ = {"comment": "平台余额退款记录表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    refund_no: Mapped[str] = mapped_column(String(32), nullable=False, comment="退款单号")
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="用户ID")
    amount: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="退款金额(分,=申请时可用余额)"
    )
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT,
        nullable=False,
        default=1,
        comment="退款状态(1:待审核;2:已退款;3:退款失败)",
    )
    channel: Mapped[str | None] = mapped_column(
        String(16), nullable=True, comment="原路退回渠道(wechat/alipay)"
    )
    channel_refund_no: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="渠道退款流水号"
    )
    apply_time: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(), comment="申请时间"
    )
    audit_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, comment="审核时间")
    auditor_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, comment="审核人ID")
    audit_remark: Mapped[str | None] = mapped_column(String(256), nullable=True, comment="审核备注")
    refund_time: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="退款完成时间"
    )
    error_message: Mapped[str | None] = mapped_column(
        String(512), nullable=True, comment="错误信息"
    )
