from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, Integer
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysUserCoupon(BaseModel):
    __tablename__ = 'sys_user_coupon'
    __table_args__ = {'comment': '用户优惠券实例表'}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='用户ID')
    coupon_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='优惠券模板ID')
    status: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=1, comment='状态(1:未使用;2:已使用;3:已过期;4:已锁定)')
    receive_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(), comment='领取时间')
    expire_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment='过期时间')
    used_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment='使用时间')
    used_order_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True, comment='使用的订单ID')
    deleted: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=0, comment='逻辑删除标识')
