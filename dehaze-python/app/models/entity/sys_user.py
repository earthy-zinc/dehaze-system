"""
用户相关实体模型
"""

from typing import Optional

from app.database import Base
from app.models.base import BaseModel
from sqlalchemy import BigInteger, Column, Index, String, Text
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column


class SysUser(BaseModel):
    __tablename__ = 'sys_user'
    __table_args__ = (
        Index('idx_sys_user_username', 'username', unique=True),
        {'comment': '用户信息表'}
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True)
    username: Mapped[Optional[str]] = mapped_column(String(64), comment='用户名')
    nickname: Mapped[Optional[str]] = mapped_column(String(64), comment='昵称')
    gender: Mapped[Optional[int]] = mapped_column(
        mysql_types.TINYINT, default=1, comment='性别((1:男')
    password: Mapped[Optional[str]] = mapped_column(String(100), comment='密码')
    dept_id: Mapped[Optional[int]] = mapped_column(BigInteger, comment='部门ID')
    avatar: Mapped[Optional[str]] = mapped_column(Text, comment='用户头像')
    mobile: Mapped[Optional[str]] = mapped_column(String(20), comment='联系方式')
    status: Mapped[Optional[int]] = mapped_column(
        mysql_types.TINYINT, default=1, comment='用户状态((1:正常')
    email: Mapped[Optional[str]] = mapped_column(String(128), comment='用户邮箱')
    deleted: Mapped[Optional[int]] = mapped_column(
        mysql_types.TINYINT, default=0, comment='逻辑删除标识(0:未删除')


class SysRole(BaseModel):
    __tablename__ = 'sys_role'
    __table_args__ = (
        Index('idx_sys_role_name', 'name', unique=True),
        {'comment': '角色表'}
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(
        String(64), nullable=False, comment='角色名称')
    code: Mapped[Optional[str]] = mapped_column(String(32), comment='角色编码')
    sort: Mapped[Optional[int]] = mapped_column(BigInteger, comment='显示顺序')
    status: Mapped[Optional[int]] = mapped_column(mysql_types.TINYINT, default=1,
                                                  comment='角色状态(1-正常；0-停用)')
    data_scope: Mapped[Optional[int]] = mapped_column(mysql_types.TINYINT,
                                                      comment='数据权限(0-所有数据；1-部门及子部门数据；2-本部门数据；3-本人数据)')
    deleted: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False,
                                         default=0, comment='逻辑删除标识(0-未删除；1-已删除)')


class SysUserRole(Base):
    """用户角色关联表（无时间字段）"""
    __tablename__ = 'sys_user_role'
    __table_args__ = {'comment': '用户和角色关联表'}

    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True,
                                         nullable=False, comment='用户ID')
    role_id: Mapped[int] = mapped_column(BigInteger, primary_key=True,
                                         nullable=False, comment='角色ID')
