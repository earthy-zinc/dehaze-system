from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin

# 权益配额覆盖的 8 类任务类型（图像处理 7 类 + 评估 1 类）
QUOTA_TASK_TYPES = [
    "dehaze",
    "derain",
    "desnow",
    "lowlight",
    "super_resolution",
    "denoise",
    "inpaint",
    "evaluate",
]

# 图像处理 7 类（不含评估），用于前端权益概览按服务类目合并
IMAGE_TASK_TYPES = [t for t in QUOTA_TASK_TYPES if t != "evaluate"]


class SysMember(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_member"
    __table_args__ = {"comment": "会员信息表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="用户ID")
    level_code: Mapped[str] = mapped_column(
        String(16), nullable=False, default="level_0", comment="会员等级"
    )
    level_source: Mapped[str] = mapped_column(
        String(16), nullable=False, default="growth", comment="等级来源"
    )
    growth_value: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, comment="成长值"
    )
    total_consumption: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, comment="累计消费金额(分)"
    )
    expire_time: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="套餐到期时间"
    )
    become_member_time: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="首次成为会员时间"
    )
    monthly_dehaze_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="本月去雾配额"
    )
    monthly_dehaze_used: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="本月已用去雾次数"
    )
    monthly_derain_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="本月去雨配额"
    )
    monthly_derain_used: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="本月已用去雨次数"
    )
    monthly_desnow_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="本月去雪配额"
    )
    monthly_desnow_used: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="本月已用去雪次数"
    )
    monthly_lowlight_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="本月低光增强配额"
    )
    monthly_lowlight_used: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="本月已用低光增强次数"
    )
    monthly_super_resolution_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="本月超分辨率配额"
    )
    monthly_super_resolution_used: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="本月已用超分辨率次数"
    )
    monthly_denoise_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="本月去噪配额"
    )
    monthly_denoise_used: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="本月已用去噪次数"
    )
    monthly_inpaint_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="本月图像修复配额"
    )
    monthly_inpaint_used: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="本月已用图像修复次数"
    )
    monthly_evaluate_quota: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="本月评估配额"
    )
    monthly_evaluate_used: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="本月已用评估次数"
    )
    quota_reset_month: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="配额所属月份(yyyyMM)"
    )
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=1, comment="状态(1:正常;0:冻结)"
    )
    frozen_reason: Mapped[str | None] = mapped_column(
        String(256), nullable=True, comment="冻结原因"
    )
    frozen_time: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="冻结时间"
    )
