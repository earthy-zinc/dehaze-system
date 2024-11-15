from datetime import datetime

from sqlalchemy import Column, String, BigInteger, Text, DateTime, SmallInteger, Integer

from base import db

class SysFile(db.Model):
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
