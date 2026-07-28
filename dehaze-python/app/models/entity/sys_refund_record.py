from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, Integer, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysRefundRecord(BaseModel):
    __tablename__ = 'sys_refund_record'
    __table_args__ = {'comment': '退款记录表'}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    refund_no: Mapped[str] = mapped_column(String(32), nullable=False, comment='退款单号')
    order_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='订单ID')
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='用户ID')
    refund_amount: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='退款金额(分)')
    reason: Mapped[str] = mapped_column(String(256), nullable=False, comment='退款原因')
    used_quota: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment='申请时已用权益次数')
    status: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=1, comment='退款状态(1:退款中;2:退款成功;3:退款失败)')
    channel: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, comment='退款渠道')
    channel_refund_no: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, comment='渠道退款流水号')
    apply_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(), comment='申请时间')
    audit_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment='审核时间')
    auditor_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True, comment='审核人ID')
    audit_remark: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, comment='审核备注')
    refund_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment='退款完成时间')
    error_message: Mapped[Optional[str]] = mapped_column(String(512), nullable=True, comment='错误信息')
    deleted: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=0, comment='逻辑删除标识')
