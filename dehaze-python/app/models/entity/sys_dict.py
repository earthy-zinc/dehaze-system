"""
字典相关实体模型
"""

from app.models.base import BaseModel
from sqlalchemy import BigInteger, Index, Integer, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column


class SysDict(BaseModel):
    __tablename__ = 'sys_dict'
    __table_args__ = {'comment': '字典数据表'}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment='主键')
    type_code: Mapped[str | None] = mapped_column(String(64), comment='字典类型编码')
    name: Mapped[str] = mapped_column(String(50), default='', comment='字典项名称')
    value: Mapped[str] = mapped_column(String(50), default='', comment='字典项值')
    sort: Mapped[int] = mapped_column(Integer, default=0, comment='排序')
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT, default=1, comment='状态(1:正常;0:禁用)')
    defaulted: Mapped[int] = mapped_column(
        mysql_types.TINYINT, default=0, comment='是否默认(1:是;0:否)')
    remark: Mapped[str] = mapped_column(String(255), default='', comment='备注')
    deleted: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=0, comment='逻辑删除标识(0:未删除;1:已删除)')


class SysDictType(BaseModel):
    __tablename__ = 'sys_dict_type'
    __table_args__ = (
        Index('type_code', 'code', unique=True),
        {'comment': '字典类型表'}
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment='主键 ')
    name: Mapped[str] = mapped_column(String(50), default='', comment='类型名称')
    code: Mapped[str] = mapped_column(String(50), default='', comment='类型编码')
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT, default=1, comment='状态(1:正常;0:禁用)')
    remark: Mapped[str | None] = mapped_column(String(255), comment='备注')
    deleted: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=0, comment='逻辑删除标识(0:未删除;1:已删除)')
