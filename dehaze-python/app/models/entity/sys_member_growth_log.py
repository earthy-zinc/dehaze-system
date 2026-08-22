from sqlalchemy import BigInteger, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysMemberGrowthLog(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_member_growth_log"
    __table_args__ = {"comment": "成长值流水表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="用户ID")
    change_type: Mapped[str] = mapped_column(String(32), nullable=False, comment="变动类型")
    change_value: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="变动值(正数增加/负数扣减)"
    )
    balance: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="变动后成长值余额")
    related_id: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="关联业务ID")
    reason: Mapped[str | None] = mapped_column(String(256), nullable=True, comment="变动原因")
    operator_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, comment="操作人ID")
