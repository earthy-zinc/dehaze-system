from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, Integer, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysMember(BaseModel):
    __tablename__ = 'sys_member'
    __table_args__ = {'comment': '会员信息表'}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='用户ID')
    level_code: Mapped[str] = mapped_column(String(16), nullable=False, default='level_0', comment='会员等级')
    level_source: Mapped[str] = mapped_column(String(16), nullable=False, default='growth', comment='等级来源')
    growth_value: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, comment='成长值')
    total_consumption: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, comment='累计消费金额(分)')
    expire_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment='套餐到期时间')
    become_member_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment='首次成为会员时间')
    monthly_dehaze_quota: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment='本月去雾配额')
    monthly_dehaze_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment='本月已用去雾次数')
    monthly_evaluate_quota: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment='本月评估配额')
    monthly_evaluate_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment='本月已用评估次数')
    quota_reset_month: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment='配额所属月份(yyyyMM)')
    status: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=1, comment='状态(1:正常;0:冻结)')
    frozen_reason: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, comment='冻结原因')
    frozen_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment='冻结时间')
    deleted: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=0, comment='逻辑删除标识')
