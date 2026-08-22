"""
部门实体模型
"""

from sqlalchemy import BigInteger, Integer, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysDept(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_dept"
    __table_args__ = {"comment": "部门表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False, default="", comment="部门名称")
    parent_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, comment="父节点id"
    )
    tree_path: Mapped[str] = mapped_column(String(255), default="", comment="父节点id路径")
    sort: Mapped[int] = mapped_column(Integer, default=0, comment="显示顺序")
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=1, comment="状态(1:正常;0:禁用)"
    )
