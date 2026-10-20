from datetime import datetime

from sqlalchemy import BigInteger, DateTime
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysUserCoupon(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_user_coupon"
    __table_args__ = {"comment": "用户优惠券实例表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="用户ID")
    coupon_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="优惠券模板ID")
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT,
        nullable=False,
        default=1,
        comment="状态(1:未使用;2:已使用;3:已过期;4:已锁定)",
    )
    receive_time: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(), comment="领取时间"
    )
    expire_time: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="过期时间"
    )
    used_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, comment="使用时间")
    used_order_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="使用的订单ID"
    )
