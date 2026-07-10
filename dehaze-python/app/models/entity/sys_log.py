"""
日志相关实体模型
"""

from datetime import datetime, timezone

from sqlalchemy import CHAR, JSON, Column, DateTime, Index, Integer, String, Text, BigInteger

from app.database import Base
from app.models.base import BaseModel


class SysPredLog(BaseModel):
    __tablename__ = 'sys_pred_log'
    __table_args__ = (
        Index('idx_algorithm_id', 'algorithm_id'),
        Index('idx_origin_md5', 'origin_md5'),
        Index('idx_pred_md5', 'pred_md5'),
        {'comment': '模型预测日志表'}
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='id')
    algorithm_id = Column(BigInteger, nullable=False, comment='算法id')
    origin_file_id = Column(BigInteger, comment='原始图像文件id（有雾图像）')
    origin_md5 = Column(CHAR(32), nullable=False, comment='原始图像md5值')
    origin_url = Column(Text, nullable=False, comment='原始图像url')
    pred_file_id = Column(BigInteger, comment='预测图像文件id')
    pred_md5 = Column(CHAR(32), nullable=False, comment='预测图像md5值')
    pred_url = Column(Text, nullable=False, comment='预测图像url')
    time = Column(Integer, default=0, comment='推理时间（秒）')


class SysEvalLog(BaseModel):
    __tablename__ = 'sys_eval_log'
    __table_args__ = (
        Index('idx_algorithm_id', 'algorithm_id'),
        Index('idx_pred_md5', 'pred_md5'),
        Index('idx_gt_md5', 'gt_md5'),
        {'comment': '模型预测日志表'}
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='id')
    algorithm_id = Column(BigInteger, nullable=False, comment='算法id')
    pred_file_id = Column(BigInteger, comment='预测图像文件id')
    pred_md5 = Column(CHAR(32), nullable=False, comment='预测图像md5值')
    pred_url = Column(Text, nullable=False, comment='预测图像url')
    gt_file_id = Column(BigInteger, comment='真值图像文件id')
    gt_md5 = Column(CHAR(32), nullable=False, comment='真值图像md5值')
    gt_url = Column(Text, nullable=False, comment='真值图像url')
    time = Column(Integer, default=0, comment='评估时间（秒）')
    result = Column(JSON, comment='预测结果')


class SysOperationLog(Base):
    """
    操作日志模型
    """
    __tablename__ = 'sys_operation_log'
    __table_args__ = (
        Index('idx_user_id', 'user_id'),
        Index('idx_create_time', 'create_time'),
        Index('idx_status', 'status'),
        {'comment': '操作日志表'}
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='日志ID')
    ip = Column(String(64), nullable=False, default='', comment='请求IP地址')
    method = Column(String(10), nullable=False, comment='请求方法(GET/POST/PUT/DELETE等)')
    path = Column(String(255), nullable=False, comment='请求路径')
    status = Column(Integer, nullable=False, default=200, comment='响应状态码')
    latency = Column(Integer, nullable=False, default=0, comment='请求耗时(毫秒)')
    agent = Column(String(512), default='', comment='用户代理(User-Agent)')
    error_message = Column(Text, default='', comment='错误信息')
    body = Column(Text, default='', comment='请求体(JSON字符串)')
    resp = Column(Text, default='', comment='响应体(JSON字符串)')
    user_id = Column(BigInteger, nullable=True, comment='用户ID')
    create_time = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment='创建时间'
    )


class SysLoginLog(Base):
    """
    登录日志模型
    """
    __tablename__ = 'sys_login_log'
    __table_args__ = (
        Index('idx_user_id', 'user_id'),
        Index('idx_create_time', 'create_time'),
        Index('idx_status', 'status'),
        {'comment': '登录日志表'}
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='日志ID')
    user_id = Column(BigInteger, nullable=True, comment='用户ID')
    username = Column(String(64), nullable=False, default='', comment='登录用户名')
    ip = Column(String(64), nullable=False, default='', comment='登录IP地址')
    location = Column(String(128), default='', comment='登录地点')
    browser = Column(String(64), default='', comment='浏览器类型')
    os = Column(String(64), default='', comment='操作系统')
    status = Column(Integer, nullable=False, default=1, comment='登录状态(1:成功;0:失败)')
    message = Column(String(255), default='', comment='登录消息')
    create_time = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment='创建时间'
    )
