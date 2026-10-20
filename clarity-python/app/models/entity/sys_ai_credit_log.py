from decimal import Decimal

from sqlalchemy import DECIMAL, BigInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import AppendOnlyModel


class SysAiCreditLog(AppendOnlyModel):
    """积分余额变动流水表(只追加，不使用逻辑删除)"""

    __tablename__ = "sys_ai_credit_log"
    __table_args__ = {"comment": "积分余额变动流水表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="用户ID(关联sys_user.id)"
    )
    source: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="变动来源(recharge;vip_gift;trial;admin_adjust;refund;consume;vip_gift_expire)",
    )
    amount: Mapped[Decimal] = mapped_column(
        DECIMAL(12, 2), nullable=False, comment="变动金额(正数增加;负数扣减)"
    )
    balance_after: Mapped[Decimal] = mapped_column(
        DECIMAL(12, 2), nullable=False, comment="变动后账户余额"
    )
    related_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="关联业务记录ID(如计费记录ID/订单ID)"
    )
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True, comment="变动原因")
    operator_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="操作人ID(NULL表示系统自动)"
    )
