from datetime import datetime

from sqlalchemy import BigInteger, DateTime, String, Text
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysPaymentRecord(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_payment_record"
    __table_args__ = {"comment": "支付流水表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    order_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="订单ID")
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="用户ID")
    payment_no: Mapped[str] = mapped_column(String(64), nullable=False, comment="支付渠道流水号")
    channel: Mapped[str] = mapped_column(String(16), nullable=False, comment="支付渠道")
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="支付金额(分)")
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=1, comment="支付状态(1:处理中;2:成功;3:失败)"
    )
    callback_time: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="回调到达时间"
    )
    callback_content: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="渠道回调原始报文"
    )
    error_message: Mapped[str | None] = mapped_column(
        String(512), nullable=True, comment="错误信息"
    )
