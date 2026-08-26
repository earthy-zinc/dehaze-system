"""AI 计费管理 - 成本管理 Schema 模型（成本单价/成本统计/对账）"""

from datetime import datetime
from decimal import Decimal

from pydantic import Field

from app.models.schema.common import OrmResult


class ModelCostDetailForm(OrmResult):
    token_type: str = Field(description="token类型(input;cached;output)")
    time_slot: str = Field(description="时段(peak;idle)")
    min_tokens: int = Field(default=0, ge=0, description="上下文分段下界")
    max_tokens: int | None = Field(default=None, description="上下文分段上界(NULL不限)")
    unit_price: Decimal = Field(ge=0, description="单价(元/百万token)")


class ModelCostCreateRequest(OrmResult):
    model_id: str = Field(..., min_length=1, max_length=64, description="模型标识")
    provider_id: int = Field(..., description="供应商ID")
    currency: str = Field(default="CNY", max_length=8, description="币种")
    effective_from: datetime | None = Field(default=None, description="价格版本生效时间")
    effective_to: datetime | None = Field(default=None, description="价格版本失效时间")
    status: int = Field(default=1, description="状态(1:生效;0:停用)")
    details: list[ModelCostDetailForm] = Field(default_factory=list, description="档位明细")


class ModelCostUpdateRequest(OrmResult):
    currency: str | None = Field(default=None, max_length=8, description="币种")
    effective_from: datetime | None = Field(default=None, description="价格版本生效时间")
    effective_to: datetime | None = Field(default=None, description="价格版本失效时间")
    status: int | None = Field(default=None, description="状态(1:生效;0:停用)")


class ModelCostDetailResult(OrmResult):
    id: int = Field(description="主键")
    price_id: int = Field(description="价格版本ID")
    token_type: str = Field(description="token类型(input;cached;output)")
    time_slot: str = Field(description="时段(peak;idle)")
    min_tokens: int = Field(description="上下文分段下界")
    max_tokens: int | None = Field(default=None, description="上下文分段上界(NULL不限)")
    unit_price: Decimal = Field(description="单价(元/百万token)")


class ModelCostResult(OrmResult):
    id: int = Field(description="主键")
    model_id: str = Field(description="模型标识")
    provider_id: int = Field(description="供应商ID")
    price_version: int = Field(description="价格版本号")
    currency: str = Field(description="币种")
    effective_from: datetime = Field(description="价格版本生效时间")
    effective_to: datetime | None = Field(default=None, description="价格版本失效时间")
    status: int = Field(description="状态(1:生效;0:停用)")
    details: list[ModelCostDetailResult] = Field(default_factory=list, description="档位明细")
    create_time: datetime | None = Field(default=None, description="创建时间")
    update_time: datetime | None = Field(default=None, description="更新时间")


class ModelCostQuery(OrmResult):
    keyword: str | None = Field(default=None, description="关键词(模型标识/供应商)")
    model_id: str | None = Field(default=None, description="模型标识")
    provider_id: int | None = Field(default=None, description="供应商ID")
    page: int = Field(default=1, ge=1, description="页码")
    size: int = Field(default=20, ge=1, le=100, description="每页数量")


class CostStatResult(OrmResult):
    dimension: str | None = Field(default=None, description="统计维度值")
    revenue: float = Field(description="收入(元)")
    cost: float = Field(description="成本(元,Σ sys_ai_billing.cost)")
    profit: float = Field(description="毛利(收入-成本)")
    profit_rate: float = Field(description="毛利率")
    metric: str = Field(description="口径(overall:整体毛利官方口径;ai:AI参考口径)")


class ReconcileImportRequest(OrmResult):
    content: str = Field(..., min_length=1, description="供应商账单内容")
    start_time: str | None = Field(default=None, description="对账周期起")
    end_time: str | None = Field(default=None, description="对账周期止")
