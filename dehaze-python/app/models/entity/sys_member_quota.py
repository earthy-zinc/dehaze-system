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
    derain_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="当月去雨配额"
    )
    derain_used: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="当月已用去雨次数"
    )
    desnow_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="当月去雪配额"
    )
    desnow_used: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="当月已用去雪次数"
    )
    lowlight_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="当月低光增强配额"
    )
    lowlight_used: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="当月已用低光增强次数"
    )
    super_resolution_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="当月超分辨率配额"
    )
    super_resolution_used: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="当月已用超分辨率次数"
    )
    denoise_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="当月去噪配额"
    )
    denoise_used: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="当月已用去噪次数"
    )
    inpaint_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="当月图像修复配额"
    )
    inpaint_used: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="当月已用图像修复次数"
    )
    evaluate_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="当月评估配额"
    )
    evaluate_used: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="当月已用评估次数"
    )
    reset_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment="配额重置时间")
