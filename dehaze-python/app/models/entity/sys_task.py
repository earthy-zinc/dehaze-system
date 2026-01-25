"""
任务实体模型
"""

from sqlalchemy import Column, DateTime, Index, Integer, String, Text, BigInteger

from app.extensions import mysql


class SysTask(mysql.Model):
    __tablename__ = 'sys_task'
    __table_args__ = (
        Index('idx_task_id', 'task_id', unique=True),
        Index('idx_status', 'status'),
        {'comment': '系统任务表'}
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='id')
    task_id = Column(String(64), nullable=False, unique=True, comment='任务ID')
    task_type = Column(String(32), nullable=False, comment='任务类型')
    status = Column(String(16), nullable=False, comment='任务状态')
    progress = Column(Integer, default=0, comment='任务进度')
    total_files = Column(Integer, comment='总文件数')
    processed_files = Column(Integer, default=0, comment='已处理文件数')
    params = Column(Text, comment='任务参数（JSON）')
    result = Column(Text, comment='任务结果（JSON）')
    error_message = Column(Text, comment='错误信息')
    created_by = Column(BigInteger, comment='创建人ID')
    created_at = Column(DateTime, comment='创建时间')
    started_at = Column(DateTime, comment='开始时间')
    completed_at = Column(DateTime, comment='完成时间')
    expires_at = Column(DateTime, comment='过期时间')
