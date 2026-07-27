from datetime import datetime
from typing import Any, Optional

from sqlalchemy import BigInteger, DateTime, JSON, String, Text
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysMessage(BaseModel):
    __tablename__ = 'sys_message'
    __table_args__ = {'comment': '消息表'}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment='主键')
    type: Mapped[str] = mapped_column(String(32), nullable=False, comment='消息类型')
    title: Mapped[str] = mapped_column(
        String(255), nullable=False, default='', comment='消息标题')
    content: Mapped[str] = mapped_column(Text, nullable=False, comment='消息正文')
    sender_type: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=1, comment='发送者类型(1:系统;2:管理员)')
    recipient_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='接收人ID')
    biz_module: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, comment='业务模块')
    biz_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, comment='业务ID')
    priority: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=2, comment='优先级(1:低;2:中;3:高;4:紧急)')
    jump_url: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, comment='跳转链接')
    extra: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True, comment='扩展数据')
    read_status: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=0, comment='已读状态(0:未读;1:已读)')
    read_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment='已读时间')
    deleted: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=0, comment='用户删除标识(0:未删除;1:已删除)')
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment='过期时间')
