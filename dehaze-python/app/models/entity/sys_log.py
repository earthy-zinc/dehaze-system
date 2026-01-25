"""
日志相关实体模型
"""

from datetime import datetime, timezone

from sqlalchemy import CHAR, JSON, Column, DateTime, Index, Integer, String, Text, BigInteger

from app.extensions import mysql


class SysPredLog(mysql.Model):
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
    create_time = Column(DateTime, nullable=False,
                         default=datetime.now(timezone.utc), comment='创建时间')
    update_time = Column(DateTime, nullable=False, default=datetime.now(
        timezone.utc), onupdate=datetime.now(timezone.utc), comment='更新时间')
    create_by = Column(BigInteger, comment='创建人ID')
    update_by = Column(BigInteger, comment='修改人ID')


class SysEvalLog(mysql.Model):
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
    create_time = Column(DateTime, nullable=False,
                         default=datetime.now(timezone.utc), comment='创建时间')
    update_time = Column(DateTime, nullable=False, default=datetime.now(
        timezone.utc), onupdate=datetime.now(timezone.utc), comment='更新时间')
    create_by = Column(BigInteger, comment='创建人ID')
    update_by = Column(BigInteger, comment='修改人ID')


class SysOperationLog(mysql.Model):
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
        default=datetime.now(timezone.utc),
        comment='创建时间'
    )

    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'ip': self.ip,
            'method': self.method,
            'path': self.path,
            'status': self.status,
            'latency': self.latency,
            'agent': self.agent,
            'errorMessage': self.error_message,
            'body': self.body,
            'resp': self.resp,
            'userId': self.user_id,
            'createTime': self.create_time.isoformat() if self.create_time else None
        }
