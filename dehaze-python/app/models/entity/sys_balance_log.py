from sqlalchemy import BigInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import AppendOnlyModel, SoftDeleteMixin


class SysBalanceLog(AppendOnlyModel, SoftDeleteMixin):
    __tablename__ = "sys_balance_log"
    __table_args__ = {"comment": "平台余额变动流水表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="用户ID")
    change_type: Mapped[str] = mapped_column(
        String(16), nullable=False, comment="变动类型(recharge/consume/refund/freeze/unfreeze)"
    )
    amount: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="变动金额(分,正数增加;负数扣减)"
    )
    balance_after: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="变动后可用余额(分)"
    )
    related_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="关联业务记录ID(如订单ID)"
    )
