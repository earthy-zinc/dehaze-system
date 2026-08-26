from sqlalchemy import BigInteger, Integer
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysBalance(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_balance"
    __table_args__ = {"comment": "平台余额账户表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="用户ID")
    balance: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, comment="可用余额(分)"
    )
    frozen_balance: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, comment="冻结余额(分)"
    )
    version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="乐观锁版本号"
    )
