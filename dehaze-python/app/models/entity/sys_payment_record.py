from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, SmallInteger, String, Text
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class SysPaymentRecord(Base):
    __tablename__ = 'sys_payment_record'
    __table_args__ = {'comment': '支付流水表'}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    order_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='订单ID')
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='用户ID')
    payment_no: Mapped[str] = mapped_column(String(64), nullable=False, comment='支付渠道流水号')
    channel: Mapped[str] = mapped_column(String(16), nullable=False, comment='支付渠道')
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='支付金额(分)')
    status: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=1, comment='支付状态(1:处理中;2:成功;3:失败)')
    callback_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment='回调到达时间')
    callback_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment='渠道回调原始报文')
    error_message: Mapped[Optional[str]] = mapped_column(String(512), nullable=True, comment='错误信息')
    deleted: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0, comment='逻辑删除标识(0:未删除;1:已删除)')
    create_time: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(), comment='创建时间')
    update_time: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(), onupdate=lambda: datetime.now(), comment='更新时间')
    create_by: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True, comment='创建人ID')
    update_by: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True, comment='修改人ID')
