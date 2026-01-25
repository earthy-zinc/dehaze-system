"""
部门实体模型
"""

from sqlalchemy import Column, DateTime, Integer, String, BigInteger
from sqlalchemy.dialects import mysql as mysql_types

from app.extensions import mysql


class SysDept(mysql.Model):
    __tablename__ = 'sys_dept'
    __table_args__ = {'comment': '部门表'}

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    name = Column(String(64), nullable=False, default='', comment='部门名称')
    parent_id = Column(BigInteger, nullable=False, default=0, comment='父节点id')
    tree_path = Column(String(255), default='', comment='父节点id路径')
    sort = Column(Integer, default=0, comment='显示顺序')
    status = Column(mysql_types.TINYINT, nullable=False,
                    default=1, comment='状态(1:正常;0:禁用)')
    deleted = Column(mysql_types.TINYINT, default=0,
                     comment='逻辑删除标识(1:已删除;0:未删除)')
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(DateTime, comment='更新时间')
    create_by = Column(BigInteger, comment='创建人ID')
    update_by = Column(BigInteger, comment='修改人ID')
