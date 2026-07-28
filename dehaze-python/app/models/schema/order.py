from typing import Optional

from pydantic import BaseModel, Field

from app.models.schema.common import BasePageQuery


class OrderCreateForm(BaseModel):
    packageId: int = Field(..., description="套餐ID")
    couponId: Optional[int] = Field(default=None, description="用户优惠券实例ID")
    payMethod: str = Field(..., description="支付方式")


class MyOrderQuery(BasePageQuery):
    status: Optional[str] = None


class OrderQuery(BasePageQuery):
    orderNo: Optional[str] = None
    keywords: Optional[str] = None
    status: Optional[str] = None
    payMethod: Optional[str] = None
    amountMin: Optional[int] = Field(default=None, ge=0)
    amountMax: Optional[int] = Field(default=None, ge=0)
    paidTimeStart: Optional[str] = None
    paidTimeEnd: Optional[str] = None


class PayRequest(BaseModel):
    payMethod: str = Field(..., description="支付方式")


class RefundApplyForm(BaseModel):
    reason: str = Field(..., min_length=1, description="退款原因")
    customReason: Optional[str] = None


class RefundAuditForm(BaseModel):
    approved: bool = Field(..., description="是否通过")
    remark: str = Field(..., description="审核备注")


class RefundQuery(BasePageQuery):
    orderNo: Optional[str] = None
    keywords: Optional[str] = None
    status: Optional[str] = None
    applyTimeStart: Optional[str] = None
    applyTimeEnd: Optional[str] = None


class AutoRenewConfigForm(BaseModel):
    packageId: int = Field(..., description="套餐ID")
    payMethod: str = Field(..., description="支付方式")
    enabled: bool = Field(..., description="是否启用")
