from datetime import datetime, time
from typing import Any, Optional

from sqlalchemy import BigInteger, DateTime, JSON, Time
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class SysNotificationSetting(Base):
    __tablename__ = 'sys_notification_setting'
    __table_args__ = {'comment': '通知偏好设置表'}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment='主键')
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='用户ID')
    push_enabled: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=1, comment='APP推送总开关(1:开;0:关)')
    dnd_enabled: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=0, comment='免打扰开关(1:开;0:关)')
    dnd_start: Mapped[Optional[time]] = mapped_column(Time, nullable=True, default=time(22, 0), comment='免打扰开始时间')
    dnd_end: Mapped[Optional[time]] = mapped_column(Time, nullable=True, default=time(8, 0), comment='免打扰结束时间')
    preferences: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True, comment='细粒度偏好')
    create_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime, server_default=None, comment='创建时间')
    update_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime, server_default=None, comment='更新时间')
