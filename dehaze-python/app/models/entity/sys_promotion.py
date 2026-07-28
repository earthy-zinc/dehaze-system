from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, String, JSON
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysPromotion(BaseModel):
    __tablename__ = 'sys_promotion'
    __table_args__ = {'comment': '促销活动表'}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    name: Mapped[str] = mapped_column(String(64), nullable=False, comment='活动名称')
    type: Mapped[str] = mapped_column(String(32), nullable=False, comment='活动类型')
    description: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, comment='活动描述')
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment='活动开始时间')
    end_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment='活动结束时间')
    activity_rules: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, comment='活动规则')
    applicable_scope: Mapped[Optional[list]] = mapped_column(JSON, nullable=True, comment='适用套餐ID列表')
    new_user_only: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=0, comment='是否新用户专享')
    status: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=1, comment='状态(1:启用;0:禁用)')
    deleted: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=0, comment='逻辑删除标识')


class SysPromotionPackage(BaseModel):
    __tablename__ = 'sys_promotion_package'
    __table_args__ = {'comment': '促销活动-套餐关联表'}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    promotion_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='促销活动ID')
    package_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='套餐ID')
    discount_type: Mapped[str] = mapped_column(String(16), nullable=False, comment='折扣类型(percent:百分比;fixed:固定金额)')
    discount_value: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, comment='折扣值')
