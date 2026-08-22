"""
用户相关实体模型
"""

from decimal import Decimal

from sqlalchemy import DECIMAL, BigInteger, Index, Integer, String, Text
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.base import BaseModel, SoftDeleteMixin


class SysUser(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_user"
    __table_args__ = (
        Index("idx_sys_user_username", "username", unique=True),
        {"comment": "用户信息表"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    username: Mapped[str | None] = mapped_column(String(64), comment="用户名")
    nickname: Mapped[str | None] = mapped_column(String(64), comment="昵称")
    gender: Mapped[int | None] = mapped_column(mysql_types.TINYINT, default=1, comment="性别((1:男;2:女;0:未知))")
    password: Mapped[str | None] = mapped_column(String(100), comment="密码")
    dept_id: Mapped[int | None] = mapped_column(BigInteger, comment="部门ID")
    avatar: Mapped[str | None] = mapped_column(Text, comment="用户头像")
    mobile: Mapped[str | None] = mapped_column(String(20), comment="联系方式")
    status: Mapped[int | None] = mapped_column(
        mysql_types.TINYINT, default=1, comment="用户状态((1:正常"
    )
    email: Mapped[str | None] = mapped_column(String(128), comment="用户邮箱")
    credits_balance: Mapped[Decimal] = mapped_column(
        DECIMAL(12, 2), nullable=False, default=Decimal("0.00"), comment="AI积分余额(充值/赠送增加;扣减减少)"
    )
    credits_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="AI积分余额乐观锁版本号"
    )


class SysRole(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_role"
    __table_args__ = (Index("idx_sys_role_name", "name", unique=True), {"comment": "角色表"})

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False, comment="角色名称")
    code: Mapped[str | None] = mapped_column(String(32), comment="角色编码")
    sort: Mapped[int | None] = mapped_column(BigInteger, comment="显示顺序")
    status: Mapped[int | None] = mapped_column(
        mysql_types.TINYINT, default=1, comment="角色状态(1-正常；0-停用)"
    )
    data_scope: Mapped[int | None] = mapped_column(
        mysql_types.TINYINT,
        comment="数据权限(0-所有数据；1-部门及子部门数据；2-本部门数据；3-本人数据)",
    )


class SysUserRole(Base):
    """用户角色关联表（无时间字段）"""

    __tablename__ = "sys_user_role"
    __table_args__ = {"comment": "用户和角色关联表"}

    user_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, nullable=False, comment="用户ID"
    )
    role_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, nullable=False, comment="角色ID"
    )
