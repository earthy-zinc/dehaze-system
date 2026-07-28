"""
日志相关实体模型
"""

from sqlalchemy import CHAR, JSON, Column, Index, Integer, String, Text, BigInteger
from sqlalchemy.dialects import mysql as mysql_types

from app.models.base import BaseModel
from app.models.enum.log_status import LogStatus


class SysPredLog(BaseModel):
    __tablename__ = 'sys_pred_log'
    __table_args__ = (
        Index('idx_algorithm_id', 'algorithm_id'),
        Index('idx_origin_md5', 'origin_md5'),
        Index('idx_pred_md5', 'pred_md5'),
        Index('idx_status', 'status'),
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
    status = Column(mysql_types.TINYINT, nullable=False, default=LogStatus.COMPLETED.value, comment='任务状态(1:处理中;2:已完成;3:失败)')
    error_message = Column(Text, nullable=True, comment='失败错误信息')


class SysEvalLog(BaseModel):
    __tablename__ = 'sys_eval_log'
    __table_args__ = (
        Index('idx_algorithm_id', 'algorithm_id'),
        Index('idx_pred_md5', 'pred_md5'),
        Index('idx_gt_md5', 'gt_md5'),
        Index('idx_status', 'status'),
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
    status = Column(mysql_types.TINYINT, nullable=False, default=LogStatus.COMPLETED.value, comment='任务状态(1:处理中;2:已完成;3:失败)')
    error_message = Column(Text, nullable=True, comment='失败错误信息')
    result = Column(JSON, comment='预测结果')
