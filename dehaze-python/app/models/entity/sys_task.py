"""
任务实体模型
"""

from datetime import datetime
from typing import Any, Optional

from app.models.base import BaseModel
from sqlalchemy import (BigInteger, DateTime, Index, Integer, JSON, String,
                        Text)
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column


class SysTask(BaseModel):
    __tablename__ = 'sys_task'
    __table_args__ = (
        Index('idx_task_id', 'task_id', unique=True),
        Index('idx_idempotency_key', 'idempotency_key', unique=True),
        Index('idx_status', 'status'),
        Index('idx_create_by_status', 'create_by', 'status'),
        {'comment': '系统任务表'}
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment='id')
    task_id: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, comment='任务ID')
    task_type: Mapped[str] = mapped_column(
        String(32), nullable=False, comment='任务类型')
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=1, comment='任务状态(1:待处理;2:处理中;3:已完成;4:失败;5:已取消)')
    progress: Mapped[int] = mapped_column(Integer, default=0, comment='任务进度')
    total_files: Mapped[Optional[int]] = mapped_column(Integer, comment='总文件数')
    processed_files: Mapped[int] = mapped_column(
        Integer, default=0, comment='已处理文件数')
    params: Mapped[Optional[str]] = mapped_column(Text, comment='任务参数（JSON）')
    result: Mapped[Optional[Any]] = mapped_column(JSON, comment='任务结果（JSON）：导出任务存储对象键（object_name），下载 URL 由响应层运行时拼接，不落库')
    error_message: Mapped[Optional[str]] = mapped_column(Text, comment='错误信息')
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, comment='开始时间')
    completed_at: Mapped[Optional[datetime]
                         ] = mapped_column(DateTime, comment='完成时间')
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, comment='过期时间')
    idempotency_key: Mapped[Optional[str]] = mapped_column(
        String(64), unique=True, comment='客户端幂等键（HTTP Idempotency-Key 头）')
    retry_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default='0', comment='MQ 重试次数')
    worker_id: Mapped[Optional[str]] = mapped_column(
        String(64), comment='执行 Worker 标识')
