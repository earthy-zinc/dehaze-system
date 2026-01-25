"""
文件相关实体模型
"""

from sqlalchemy import CHAR, VARCHAR, Column, DateTime, Index, Integer, String, Text, BigInteger

from app.extensions import mysql


class SysFile(mysql.Model):
    __tablename__ = 'sys_file'
    __table_args__ = (
        Index('md5_key', 'md5', unique=True),
        {'comment': '文件表'}
    )

    id = Column(Integer, primary_key=True, autoincrement=True, comment='文件id')
    type = Column(String(100), nullable=True, comment='文件类型')
    url = Column(Text, nullable=True, comment='文件url')
    name = Column(String(100), nullable=False, comment='文件原始名')
    object_name = Column(String(100), nullable=False, comment='文件存储名')
    size = Column(String(100), nullable=False, default='0', comment='文件大小')
    path = Column(String(255), nullable=False, comment='文件路径')
    md5 = Column(CHAR(32), nullable=False, unique=True,
                 comment='文件的MD5值，用于比对文件是否相同')
    create_time = Column(DateTime, nullable=False, comment='创建时间')
    update_time = Column(DateTime, nullable=True, comment='更新时间')


class SysWpxFile(mysql.Model):
    __tablename__ = 'sys_wpx_file'
    __table_args__ = {'comment': 'WPX文件表'}

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='id')
    origin_file_id = Column(BigInteger, comment='旧文件id')
    origin_md5 = Column(CHAR(32), unique=True,
                        nullable=False, comment='旧文件的MD5值')
    origin_path = Column(VARCHAR(255), nullable=False, comment='旧文件路径')
    new_file_id = Column(BigInteger, comment='新文件id')
    new_path = Column(VARCHAR(255), nullable=False, comment='新文件路径')
    new_md5 = Column(CHAR(32), unique=True, nullable=False, comment='新文件的MD5值')
