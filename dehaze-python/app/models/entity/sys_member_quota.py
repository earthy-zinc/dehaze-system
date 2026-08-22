from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysMemberQuota(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_member_quota"
    __table_args__ = {"comment": "会员月度配额历史表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="用户ID")
    quota_month: Mapped[int] = mapped_column(Integer, nullable=False, comment="配额月份(yyyyMM)")
    level_code: Mapped[str] = mapped_column(String(16), nullable=False, comment="当月会员等级")
    dehaze_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="当月去雾配额"
    )
    dehaze_used: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="当月已用去雾次数"
    )
    evaluate_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="当月评估配额"
    )
    evaluate_used: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="当月已用评估次数"
    )
    reset_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment="配额重置时间")
