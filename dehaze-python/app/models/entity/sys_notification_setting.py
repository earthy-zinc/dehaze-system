from datetime import datetime, time
from typing import Any, Optional

from sqlalchemy import BigInteger, DateTime, JSON, SmallInteger, Time
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
    deleted: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0, comment='逻辑删除标识(0:未删除;1:已删除)')
    create_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, comment='创建时间')
    update_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now, comment='更新时间')
    create_by: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True, comment='创建人ID')
    update_by: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True, comment='修改人ID')
