from datetime import time
from typing import Any

from sqlalchemy import JSON, BigInteger, Time
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysNotificationSetting(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_notification_setting"
    __table_args__ = {"comment": "通知偏好设置表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="用户ID")
    push_enabled: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=1, comment="APP推送总开关(1:开;0:关)"
    )
    dnd_enabled: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=0, comment="免打扰开关(1:开;0:关)"
    )
    dnd_start: Mapped[time | None] = mapped_column(
        Time, nullable=True, default=time(22, 0), comment="免打扰开始时间"
    )
    dnd_end: Mapped[time | None] = mapped_column(
        Time, nullable=True, default=time(8, 0), comment="免打扰结束时间"
    )
    preferences: Mapped[Any | None] = mapped_column(JSON, nullable=True, comment="细粒度偏好")
