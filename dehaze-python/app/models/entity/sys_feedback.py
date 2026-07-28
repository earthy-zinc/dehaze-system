from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, String, JSON
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysFeedback(BaseModel):
    __tablename__ = 'sys_feedback'
    __table_args__ = {'comment': '用户反馈表'}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='提交用户ID')
    feedback_type: Mapped[str] = mapped_column(String(32), nullable=False, comment='反馈类型')
    title: Mapped[str] = mapped_column(String(50), nullable=False, comment='反馈标题')
    content: Mapped[str] = mapped_column(String(1000), nullable=False, comment='反馈内容')
    contact: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, comment='联系方式')
    images: Mapped[Optional[list]] = mapped_column(JSON, nullable=True, comment='截图URL（JSON数组）')
    related_module: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, comment='相关模块')
    status: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=1, comment='状态(1:待处理;2:处理中;3:已回复;4:已关闭)')
    priority: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=1, comment='优先级(1:普通;2:紧急;3:高优)')
    assignee_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True, comment='处理人ID')
    assigned_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment='分配时间')
    tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True, comment='反馈标签（JSON数组）')
    close_reason: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, comment='关闭原因')
    deleted: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=0, comment='逻辑删除标识')
