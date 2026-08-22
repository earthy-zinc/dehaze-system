from datetime import datetime

from sqlalchemy import JSON, BigInteger, DateTime, Integer, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysCoupon(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_coupon"
    __table_args__ = {"comment": "优惠券模板表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False, comment="优惠券名称")
    type: Mapped[str] = mapped_column(String(32), nullable=False, comment="类型")
    face_value: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, comment="面值")
    threshold: Mapped[int | None] = mapped_column(BigInteger, nullable=True, comment="使用门槛(分)")
    valid_type: Mapped[str] = mapped_column(String(16), nullable=False, comment="有效期类型")
    valid_start: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="有效期开始时间"
    )
    valid_end: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="有效期结束时间"
    )
    valid_days: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="领取后有效天数")
    total_qty: Mapped[int] = mapped_column(
        Integer, nullable=False, default=-1, comment="发放总量(-1为不限)"
    )
    issued_qty: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="已发放数量"
    )
    used_qty: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="已使用数量")
    per_user_limit: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, comment="每人限领数量"
    )
    applicable_scope: Mapped[list | None] = mapped_column(
        JSON, nullable=True, comment="适用套餐ID列表"
    )
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=1, comment="状态(1:启用;0:禁用)"
    )
