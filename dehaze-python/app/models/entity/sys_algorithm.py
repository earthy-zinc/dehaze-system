"""
算法模型实体
对齐 dehaze-java SysAlgorithm.java + BaseEntity
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, Integer, String, Text
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysAlgorithm(BaseModel):
    """算法模型表 (对齐 Java SysAlgorithm + BaseEntity)"""
    __tablename__ = 'sys_algorithm'
    __table_args__ = {'comment': '算法模型表'}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True,
                                    autoincrement=True, comment='模型id')
    parent_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, default=0, comment='模型的父id')
    type: Mapped[str] = mapped_column(String(100), default='', comment='模型类型')
    name: Mapped[str] = mapped_column(
        String(64), nullable=False, comment='模型名称')
    # 版本号 (对齐 Java schema.sql: varchar(50))
    version: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, comment='算法版本号')
    img: Mapped[Optional[str]] = mapped_column(Text, comment='模型图片')
    path: Mapped[str] = mapped_column(
        String(255), default='', comment='模型存储路径')
    size: Mapped[Optional[str]] = mapped_column(String(100), comment='模型大小')
    params: Mapped[Optional[str]] = mapped_column(String(255), comment='模型参数')
    flops: Mapped[Optional[str]] = mapped_column(
        String(255), comment='模型浮点运算次数')
    import_path: Mapped[Optional[str]] = mapped_column(
        String(255), comment='模型代码导入路径')
    description: Mapped[Optional[str]] = mapped_column(
        String(2048), comment='针对该模型的详细描述')
    # 状态机：0=草稿 1=测试中 2=待审核 3=已发布 4=已停用 5=已归档
    # (Java SysAlgorithm.status 使用 Integer, 支持状态机)
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT, default=0, comment='状态(0:草稿;1:测试中;2:待审核;3:已发布;4:已停用;5:已归档)')
    # 审核字段 (对齐 Java schema.sql: audit_by bigint, audit_remark varchar(500))
    audit_by: Mapped[Optional[int]] = mapped_column(
        BigInteger, comment='审核人ID')
    audit_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime, comment='审核时间')
    audit_remark: Mapped[Optional[str]] = mapped_column(
        String(500), comment='审核备注')


class SysAlgorithmVersion(BaseModel):
    """算法版本历史表 (对齐 Java SysAlgorithmVersion.java + BaseEntity)

    字段完全对齐 config/sql/schema.sql 中 sys_algorithm_version 表定义.
    """
    __tablename__ = 'sys_algorithm_version'
    __table_args__ = {'comment': '算法版本历史表'}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    algorithm_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='关联算法ID')
    version: Mapped[str] = mapped_column(String(50), nullable=False, comment='版本号')
    change_log: Mapped[Optional[str]] = mapped_column(Text, comment='变更日志')
    status: Mapped[Optional[int]] = mapped_column(Integer, comment='该版本时的状态')
    config_json: Mapped[Optional[str]] = mapped_column(Text, comment='该版本时的配置JSON')
    model_file_id: Mapped[Optional[int]] = mapped_column(BigInteger, comment='模型文件ID')
    is_active: Mapped[Optional[int]] = mapped_column(
        mysql_types.TINYINT(1), default=0, comment='是否当前活跃版本')
