from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, Integer, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysOrder(BaseModel):
    __tablename__ = 'sys_order'
    __table_args__ = {'comment': '订单表'}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    order_no: Mapped[str] = mapped_column(String(32), nullable=False, comment='订单号')
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='用户ID')
    package_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='套餐ID')
    package_name: Mapped[str] = mapped_column(String(32), nullable=False, comment='套餐名称')
    package_level: Mapped[str] = mapped_column(String(16), nullable=False, comment='套餐对应会员等级')
    period_days: Mapped[int] = mapped_column(Integer, nullable=False, comment='有效期天数')
    original_price: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='原价(分)')
    discount_amount: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, comment='促销折扣金额(分)')
    coupon_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True, comment='用户优惠券实例ID')
    coupon_amount: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, comment='优惠券抵扣金额(分)')
    payable_amount: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='应付金额(分)')
    paid_amount: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, comment='实付金额(分)')
    pay_method: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, comment='支付方式')
    status: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=1, comment='订单状态(1:待支付;2:已支付;3:已完成;4:已取消;5:退款中;6:已退款)')
    expire_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment='支付超时时间')
    effective_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment='权益生效时间')
    package_expire_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment='套餐到期时间')
    paid_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment='支付成功时间')
    cancel_reason: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, comment='取消原因')
    is_auto_renew: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=0, comment='是否自动续费订单')
    deleted: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=0, comment='逻辑删除标识')
