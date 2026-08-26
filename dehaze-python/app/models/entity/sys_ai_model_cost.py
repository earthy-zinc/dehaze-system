from datetime import datetime
from decimal import Decimal

from sqlalchemy import BigInteger, DateTime, Integer, Numeric, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysAiModelCost(BaseModel, SoftDeleteMixin):
    """AI模型成本单价版本主表（供应商采购价，单价以档位明细行表达）"""

    __tablename__ = "sys_ai_model_cost"
    __table_args__ = {"comment": "AI模型成本单价版本表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    model_id: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="模型标识(关联sys_ai_model.model_id)"
    )
    provider_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="供应商ID")
    price_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, comment="价格版本号(同模型同供应商内递增)"
    )
    currency: Mapped[str] = mapped_column(
        String(8), nullable=False, default="CNY", comment="计费币种(CNY/USD)"
    )
    effective_from: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment="价格版本生效时间")
    effective_to: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="价格版本失效时间(NULL表示当前版本)"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="状态(1:生效;0:停用)"
    )


class SysAiModelCostDetail(BaseModel, SoftDeleteMixin):
    """AI模型成本单价档位明细表（token类型 × 上下文分段 × 时段）"""

    __tablename__ = "sys_ai_model_cost_detail"
    __table_args__ = {"comment": "AI模型成本单价档位明细表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    price_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, index=True, comment="价格版本ID(关联sys_ai_model_cost.id)"
    )
    token_type: Mapped[str] = mapped_column(
        String(16), nullable=False, comment="token类型(input;cached;output)"
    )
    time_slot: Mapped[str] = mapped_column(
        String(16), nullable=False, comment="时段档位(peak;idle)"
    )
    min_tokens: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, comment="上下文分段下界"
    )
    max_tokens: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="上下文分段上界(NULL不限)"
    )
    unit_price: Mapped[Decimal] = mapped_column(
        Numeric(12, 4), nullable=False, default=0, comment="单价(元/百万token)"
    )
