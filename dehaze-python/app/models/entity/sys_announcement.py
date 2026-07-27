from datetime import datetime
from typing import Any, Optional

from sqlalchemy import BigInteger, DateTime, JSON, String, Text
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysAnnouncement(BaseModel):
    __tablename__ = 'sys_announcement'
    __table_args__ = {'comment': '系统公告表'}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment='主键')
    title: Mapped[str] = mapped_column(
        String(255), nullable=False, default='', comment='公告标题')
    content: Mapped[str] = mapped_column(Text, nullable=False, comment='公告内容')
    type: Mapped[str] = mapped_column(
        String(32), nullable=False, default='operation', comment='公告类型')
    importance: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=1, comment='重要级别(1:普通;2:重要)')
    target_scope: Mapped[str] = mapped_column(
        String(32), nullable=False, default='all', comment='发送范围')
    target_params: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True, comment='范围参数')
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=1, comment='公告状态(1:草稿;2:待发送;3:已发送;4:已取消)')
    send_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment='发送时间')
    expire_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment='过期时间')
    sent_count: Mapped[Optional[int]] = mapped_column(nullable=False, default=0, comment='已发送人数')
    deleted: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=0, comment='逻辑删除标识')
