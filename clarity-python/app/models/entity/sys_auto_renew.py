from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysAutoRenew(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_auto_renew"
    __table_args__ = {"comment": "自动续费配置表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="用户ID")
    package_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="套餐ID")
    pay_method: Mapped[str] = mapped_column(String(16), nullable=False, comment="支付方式")
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=1, comment="状态(1:启用;0:已关闭)"
    )
    next_renew_time: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="下次扣款时间"
    )
    fail_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="连续失败次数"
    )
    last_renew_order_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="上次续费订单ID"
    )
    close_reason: Mapped[str | None] = mapped_column(String(256), nullable=True, comment="关闭原因")
