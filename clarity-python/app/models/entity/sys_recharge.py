from datetime import datetime

from sqlalchemy import BigInteger, DateTime, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysRecharge(BaseModel, SoftDeleteMixin):
    """余额充值订单（人民币余额账户充值，与积分卡 sys_order 交易隔离）"""

    __tablename__ = "sys_recharge"
    __table_args__ = {"comment": "余额充值订单表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    recharge_no: Mapped[str] = mapped_column(String(32), nullable=False, comment="充值单号")
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="用户ID")
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="充值金额(分)")
    pay_method: Mapped[str] = mapped_column(
        String(16), nullable=False, comment="支付方式(wechat:微信;alipay:支付宝)"
    )
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT,
        nullable=False,
        default=1,
        comment="充值状态(1:待支付;2:已支付;3:已关闭)",
    )
    channel_payment_no: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="渠道支付流水号(唯一,回调幂等依据)"
    )
    pay_time: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="支付成功时间"
    )
