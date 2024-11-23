from datetime import datetime

from flask import current_app
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, String, BigInteger, Text, DateTime, SmallInteger, Integer

mysql: SQLAlchemy = current_app.extensions['mysql']

class SysFile(mysql.Model):
    __tablename__ = 'sys_file'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='文件id')
    type = Column(String(100), nullable=True, comment='文件类型')
    url = Column(String(255), nullable=True, comment='文件url')
    name = Column(String(100), nullable=False, comment='文件原始名')
    object_name = Column(String(100), nullable=False, comment='文件存储名')
    size = Column(String(100), nullable=False, default='0', comment='文件大小')
    path = Column(String(255), nullable=False, comment='文件路径')
    md5 = Column(String(32), unique=True, nullable=False, comment='文件的MD5值，用于比对文件是否相同')
    create_time = Column(DateTime, nullable=False, default=datetime.utcnow, comment='创建时间')
    update_time = Column(DateTime, nullable=True, default=datetime.utcnow, onupdate=datetime.utcnow, comment='更新时间')

class SysAlgorithm(mysql.Model):
    __tablename__ = 'sys_algorithm'

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='模型id')
    parent_id = Column(BigInteger, default=0, comment='模型的父id')
    type = Column(String(100), default='', comment='模型类型')
    name = Column(String(64), nullable=False, comment='模型名称')
    path = Column(String(255), default='', comment='模型存储路径')
    size = Column(String(100), default=None, comment='模型大小')
    import_path = Column(String(255), default='', comment='模型代码导入路径')
    description = Column(Text, default=None, comment='针对该模型的详细描述')
    status = Column(SmallInteger, default=1, comment='状态(1:启用；0:禁用)')
    create_time = Column(DateTime, default=datetime.utcnow, comment='创建时间')
    update_time = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment='更新时间')
    create_by = Column(BigInteger, nullable=True, comment='创建人ID')
    update_by = Column(BigInteger, nullable=True, comment='修改人ID')
