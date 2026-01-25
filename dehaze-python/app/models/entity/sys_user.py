"""
用户相关实体模型
"""

from sqlalchemy import Column, DateTime, Index, String, Text, BigInteger
from sqlalchemy.dialects import mysql as mysql_types

from app.extensions import mysql


class SysUser(mysql.Model):
    __tablename__ = 'sys_user'
    __table_args__ = (
        Index('idx_sys_user_username', 'username', unique=True),
        {'comment': '用户信息表'}
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    username = Column(String(64), comment='用户名')
    nickname = Column(String(64), comment='昵称')
    gender = Column(mysql_types.TINYINT, default=1, comment='性别((1:男')
    password = Column(String(100), comment='密码')
    dept_id = Column(BigInteger, comment='部门ID')
    avatar = Column(Text, comment='用户头像')
    mobile = Column(String(20), comment='联系方式')
    status = Column(mysql_types.TINYINT, default=1, comment='用户状态((1:正常')
    email = Column(String(128), comment='用户邮箱')
    deleted = Column(mysql_types.TINYINT, default=0, comment='逻辑删除标识(0:未删除')
    create_time = Column(DateTime)
    update_time = Column(DateTime)


class SysRole(mysql.Model):
    __tablename__ = 'sys_role'
    __table_args__ = (
        Index('idx_sys_role_name', 'name', unique=True),
        {'comment': '角色表'}
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(64), nullable=False, comment='角色名称')
    code = Column(String(32), comment='角色编码')
    sort = Column(BigInteger, comment='显示顺序')
    status = Column(mysql_types.TINYINT, default=1,
                    comment='角色状态(1-正常；0-停用)')
    data_scope = Column(mysql_types.TINYINT,
                        comment='数据权限(0-所有数据；1-部门及子部门数据；2-本部门数据；3-本人数据)')
    deleted = Column(mysql_types.TINYINT, nullable=False,
                     default=0, comment='逻辑删除标识(0-未删除；1-已删除)')
    create_time = Column(DateTime)
    update_time = Column(DateTime)


class SysUserRole(mysql.Model):
    __tablename__ = 'sys_user_role'
    __table_args__ = {'comment': '用户和角色关联表'}

    user_id = Column(BigInteger, primary_key=True,
                     nullable=False, comment='用户ID')
    role_id = Column(BigInteger, primary_key=True,
                     nullable=False, comment='角色ID')
