"""
任务实体模型
"""

from datetime import datetime
from typing import Optional

from app.database import Base
from sqlalchemy import (BigInteger, Column, DateTime, Index, Integer, String,
                        Text)
from sqlalchemy.orm import Mapped, mapped_column


class SysTask(Base):
    __tablename__ = 'sys_task'
    __table_args__ = (
        Index('idx_task_id', 'task_id', unique=True),
        Index('idx_status', 'status'),
        Index('idx_created_by_status', 'created_by', 'status'),
        {'comment': '系统任务表'}
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment='id')
    task_id: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, comment='任务ID')
    task_type: Mapped[str] = mapped_column(
        String(32), nullable=False, comment='任务类型')
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, comment='任务状态')
    progress: Mapped[int] = mapped_column(Integer, default=0, comment='任务进度')
    total_files: Mapped[Optional[int]] = mapped_column(Integer, comment='总文件数')
    processed_files: Mapped[int] = mapped_column(
        Integer, default=0, comment='已处理文件数')
    params: Mapped[Optional[str]] = mapped_column(Text, comment='任务参数（JSON）')
    result: Mapped[Optional[str]] = mapped_column(Text, comment='任务结果（JSON）')
    error_message: Mapped[Optional[str]] = mapped_column(Text, comment='错误信息')
    created_by: Mapped[Optional[int]] = mapped_column(
        BigInteger, comment='创建人ID')
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, comment='创建时间')
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, comment='更新时间')
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, comment='开始时间')
    completed_at: Mapped[Optional[datetime]
                         ] = mapped_column(DateTime, comment='完成时间')
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, comment='过期时间')
