from typing import Literal

from pydantic import BaseModel, Field


class BenefitOverrides(BaseModel):
    monthlyDehazeQuota: int | None = None
    monthlyEvaluateQuota: int | None = None
    historyRetention: int | None = None
    batchLimit: int | None = None
    priority: int | None = None
    advancedParams: int | None = None
    hdExport: int | None = None
    reportExport: int | None = None
    batchDownload: int | None = None


class PackageForm(BaseModel):
    id: int | None = None
    name: str = Field(..., min_length=2, max_length=32, description="套餐名称")
    packageType: Literal["vip", "credit"] = Field(default="vip", description="商品类型")
    creditAmount: int | None = Field(default=None, ge=1, description="可得积分(积分卡必填)")
    levelCode: str | None = None
    period: Literal["monthly", "quarterly", "yearly"] | None = None
    periodDays: int | None = Field(default=None, ge=1, le=365, description="有效期天数(积分卡为NULL)")
    originalPrice: int = Field(..., ge=1, description="原价(分)")
    salePrice: int = Field(..., ge=1, description="促销价(分)")
    description: str | None = Field(default=None, max_length=256, description="套餐描述")
    benefitOverrides: BenefitOverrides | None = None
    sort: int | None = Field(default=0, ge=0, le=999, description="排序值")
    status: int | None = Field(default=0, ge=0, le=1)


class CouponForm(BaseModel):
    id: int | None = None
    name: str = Field(..., min_length=1, description="优惠券名称")
    type: str = Field(..., description="类型")
    faceValue: int = Field(..., ge=0, description="面值")
    threshold: int | None = Field(default=None, ge=0, description="使用门槛(分)")
    validType: Literal["fixed", "relative"] = Field(..., description="有效期类型")
    validStart: str | None = None
    validEnd: str | None = None
    validDays: int | None = Field(default=None, ge=1, description="领取后有效天数")
    totalQty: int = Field(..., ge=0, description="发放总量")
    perUserLimit: int = Field(..., ge=1, description="每人限领数量")
    applicableScope: list[int | str] | None = Field(
        default=None, description="适用商品(商品ID或商品类型vip/credit，NULL全部适用)"
    )
    status: int | None = Field(default=1, ge=0, le=1)


class CouponBatchDistributeForm(BaseModel):
    couponId: int = Field(..., description="优惠券ID")
    targetScope: str = Field(..., description="发放范围(all/level/users)")
    levelCodes: list[str] | None = None
    userIds: list[int] | None = None
