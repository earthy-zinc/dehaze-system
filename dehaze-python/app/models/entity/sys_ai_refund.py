from sqlalchemy import BigInteger, Integer, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import AppendOnlyModel


class SysAiRefund(AppendOnlyModel):
    """AI计费退款申请表(只追加，不使用逻辑删除)"""

    __tablename__ = "sys_ai_refund"
    __table_args__ = {"comment": "AI计费退款申请表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="用户ID(关联sys_user.id)"
    )
    billing_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="原计费记录ID(关联sys_ai_billing.id)"
    )
    amount: Mapped[int] = mapped_column(Integer, nullable=False, comment="退款积分数")
    reason: Mapped[str] = mapped_column(String(255), nullable=False, comment="退款原因")
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="退款状态(1:待审核;2:已通过;3:已驳回)"
    )
    create_by: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="申请人ID(用户申请退款时记录)"
    )
    auditor_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, comment="审核人ID")
    audit_remark: Mapped[str | None] = mapped_column(String(255), nullable=True, comment="审核意见")
