"""
字典相关实体模型
"""

from sqlalchemy import Column, DateTime, Index, Integer, String, BigInteger
from sqlalchemy.dialects import mysql as mysql_types

from app.extensions import mysql


class SysDict(mysql.Model):
    __tablename__ = 'sys_dict'
    __table_args__ = {'comment': '字典数据表'}

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    type_code = Column(String(64), comment='字典类型编码')
    name = Column(String(50), default='', comment='字典项名称')
    value = Column(String(50), default='', comment='字典项值')
    sort = Column(Integer, default=0, comment='排序')
    status = Column(mysql_types.TINYINT, default=0, comment='状态(1:正常;0:禁用)')
    defaulted = Column(mysql_types.TINYINT, default=0, comment='是否默认(1:是;0:否)')
    remark = Column(String(255), default='', comment='备注')
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(DateTime, comment='更新时间')


class SysDictType(mysql.Model):
    __tablename__ = 'sys_dict_type'
    __table_args__ = (
        Index('type_code', 'code', unique=True),
        {'comment': '字典类型表'}
    )

    id = Column(BigInteger, primary_key=True,
                autoincrement=True, comment='主键 ')
    name = Column(String(50), default='', comment='类型名称')
    code = Column(String(50), default='', comment='类型编码')
    status = Column(mysql_types.TINYINT, default=0, comment='状态(0:正常;1:禁用)')
    remark = Column(String(255), comment='备注')
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(DateTime, comment='更新时间')
