"""
算法模型实体
"""

from sqlalchemy import Column, DateTime, String, Text, BigInteger
from sqlalchemy.dialects import mysql as mysql_types

from app.extensions import mysql


class SysAlgorithm(mysql.Model):
    __tablename__ = 'sys_algorithm'
    __table_args__ = {'comment': '算法模型表'}

    id = Column(BigInteger, primary_key=True,
                autoincrement=True, comment='模型id')
    parent_id = Column(BigInteger, default=0, comment='模型的父id')
    type = Column(String(100), default='', comment='模型类型')
    name = Column(String(64), nullable=False, comment='模型名称')
    img = Column(Text, comment='模型图片')
    path = Column(String(255), default='', comment='模型存储路径')
    size = Column(String(100), comment='模型大小')
    params = Column(String(255), comment='模型参数')
    flops = Column(String(255), comment='模型浮点运算次数')
    import_path = Column(String(255), comment='模型代码导入路径')
    description = Column(String(2048), comment='针对该模型的详细描述')
    status = Column(mysql_types.TINYINT, default=1, comment='状态(1:启用；0:禁用)')
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(DateTime, comment='更新时间')
    create_by = Column(BigInteger, comment='创建人ID')
    update_by = Column(BigInteger, comment='修改人ID')
