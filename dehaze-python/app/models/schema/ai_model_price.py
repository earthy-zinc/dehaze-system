"""AI 模型管理 - 用户售价 Schema 模型（价格版本化 + 档位明细 + 积分换算）"""

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import Field

from app.models.schema.common import OrmResult

# 计费计量单位白名单（token 模型 / 张 / 秒 / 万字符 / 帧，对齐计费统一口径）
BillingUnit = Literal["million_tokens", "image", "second", "ten_thousand_chars", "frame"]
# 档位计量类型：token 模型按 token 档；非 token 模型单行 unit 档（不设峰谷）
UsageKind = Literal["token", "unit"]

_BILLING_UNIT_DESC = (
    "计费计量单位(million_tokens:百万token;image:张;second:秒;ten_thousand_chars:万字符;frame:帧)"
)


class ModelPriceDetailForm(OrmResult):
    token_type: str = Field(
        description="token类型(token档为input;cached;output;unit档为unit,与usage_kind联动)"
    )
    usage_kind: UsageKind = Field(
        default="token", description="计量类型(token:token计价;unit:按计量单位整档计价)"
    )
    time_slot: str = Field(description="时段档位(peak;idle)")
    min_tokens: int = Field(default=0, ge=0, description="上下文分段下界")
    max_tokens: int | None = Field(default=None, description="上下文分段上界(NULL不限)")
    unit_price: Decimal = Field(ge=0, description="单价(积分/百万token)")


class ModelPriceCreateRequest(OrmResult):
    model_id: str = Field(..., min_length=1, max_length=64, description="模型标识")
    provider_id: int = Field(..., description="供应商ID")
    unit: str = Field(
        default="credits_per_million",
        max_length=24,
        description="单价单位(credits_per_million:积分/百万token)",
    )
    billing_unit: BillingUnit = Field(default="million_tokens", description=_BILLING_UNIT_DESC)
    effective_from: datetime | None = Field(default=None, description="价格版本生效时间")
    effective_to: datetime | None = Field(default=None, description="价格版本失效时间")
    status: int = Field(default=1, description="状态(1:生效;0:停用)")
    details: list[ModelPriceDetailForm] = Field(default_factory=list, description="档位明细")


class ModelPriceUpdateRequest(OrmResult):
    unit: str | None = Field(default=None, max_length=24, description="单价单位")
    billing_unit: BillingUnit | None = Field(default=None, description="计费计量单位")
    effective_from: datetime | None = Field(default=None, description="价格版本生效时间")
    effective_to: datetime | None = Field(default=None, description="价格版本失效时间")
    status: int | None = Field(default=None, description="状态(1:生效;0:停用)")


class ModelPriceDetailResult(OrmResult):
    id: int = Field(description="主键")
    price_id: int = Field(description="价格版本ID")
    token_type: str = Field(
        description="token类型(token档为input;cached;output;unit档为unit,与usage_kind联动)"
    )
    usage_kind: UsageKind = Field(description="计量类型(token:token计价;unit:按计量单位整档计价)")
    time_slot: str = Field(description="时段档位(peak;idle)")
    min_tokens: int = Field(description="上下文分段下界")
    max_tokens: int | None = Field(default=None, description="上下文分段上界(NULL不限)")
    unit_price: Decimal = Field(description="单价(积分/百万token)")


class ModelPriceResult(OrmResult):
    id: int = Field(description="主键")
    model_id: str = Field(description="模型标识")
    provider_id: int = Field(description="供应商ID")
    price_version: int = Field(description="价格版本号")
    unit: str = Field(description="单价单位")
    billing_unit: BillingUnit = Field(description="计费计量单位")
    effective_from: datetime = Field(description="价格版本生效时间")
    effective_to: datetime | None = Field(default=None, description="价格版本失效时间")
    status: int = Field(description="状态(1:生效;0:停用)")
    details: list[ModelPriceDetailResult] = Field(default_factory=list, description="档位明细")
    create_time: datetime | None = Field(default=None, description="创建时间")
    update_time: datetime | None = Field(default=None, description="更新时间")


class ModelPriceQuery(OrmResult):
    model_id: str | None = Field(default=None, description="模型标识")
    provider_id: int | None = Field(default=None, description="供应商ID")
    page: int = Field(default=1, ge=1, description="页码")
    size: int = Field(default=20, ge=1, le=100, description="每页数量")


class ModelPriceCalculateResult(OrmResult):
    credits: int = Field(description="换算积分(四舍五入取整,单次至少1积分)")
