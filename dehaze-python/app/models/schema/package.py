from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.models.schema.common import BasePageQuery


class PackageQuery(BasePageQuery):
    name: Optional[str] = None
    levelCode: Optional[str] = None
    period: Optional[str] = None
    status: Optional[int] = Field(default=None, ge=0, le=1)
    startTime: Optional[str] = None
    endTime: Optional[str] = None


class BenefitOverrides(BaseModel):
    monthlyDehazeQuota: Optional[int] = None
    monthlyEvaluateQuota: Optional[int] = None
    historyRetention: Optional[int] = None
    batchLimit: Optional[int] = None
    priority: Optional[int] = None
    advancedParams: Optional[int] = None
    hdExport: Optional[int] = None
    reportExport: Optional[int] = None
    batchDownload: Optional[int] = None


class PackageForm(BaseModel):
    id: Optional[int] = None
    name: str = Field(..., min_length=1, description="套餐名称")
    levelCode: str = Field(..., description="会员等级")
    period: str = Field(..., description="计费周期")
    periodDays: int = Field(..., ge=1, description="有效期天数")
    originalPrice: int = Field(..., ge=0, description="原价(分)")
    salePrice: int = Field(..., ge=0, description="促销价(分)")
    description: Optional[str] = None
    benefitOverrides: Optional[BenefitOverrides] = None
    sort: Optional[int] = 0
    status: Optional[int] = Field(default=0, ge=0, le=1)


class CouponForm(BaseModel):
    id: Optional[int] = None
    name: str = Field(..., min_length=1, description="优惠券名称")
    type: str = Field(..., description="类型")
    faceValue: int = Field(..., ge=0, description="面值")
    threshold: Optional[int] = Field(default=None, ge=0, description="使用门槛(分)")
    validType: str = Field(..., description="有效期类型")
    validStart: Optional[str] = None
    validEnd: Optional[str] = None
    validDays: Optional[int] = Field(default=None, ge=1, description="领取后有效天数")
    totalQty: int = Field(..., ge=0, description="发放总量")
    perUserLimit: int = Field(..., ge=1, description="每人限领数量")
    applicableScope: Optional[list[int]] = None
    status: Optional[int] = Field(default=1, ge=0, le=1)


class CouponQuery(BasePageQuery):
    name: Optional[str] = None
    type: Optional[str] = None
    status: Optional[int] = Field(default=None, ge=0, le=1)


class CouponBatchDistributeForm(BaseModel):
    couponId: int = Field(..., description="优惠券ID")
    targetScope: str = Field(..., description="发放范围(all/level/users)")
    levelCodes: Optional[list[str]] = None
    userIds: Optional[list[int]] = None
