from pydantic import BaseModel, Field


class OrderCreateForm(BaseModel):
    packageId: int = Field(..., description="套餐ID")
    couponId: int | None = Field(default=None, description="用户优惠券实例ID")
    payMethod: str = Field(..., description="支付方式")


class PayRequest(BaseModel):
    payMethod: str = Field(..., description="支付方式")


class RefundApplyForm(BaseModel):
    reason: str = Field(..., min_length=1, description="退款原因")
    customReason: str | None = None


class RefundAuditForm(BaseModel):
    approved: bool = Field(..., description="是否通过")
    remark: str = Field(..., description="审核备注")


class AutoRenewConfigForm(BaseModel):
    packageId: int = Field(..., description="套餐ID")
    payMethod: str = Field(..., description="支付方式")
    enabled: bool = Field(..., description="是否启用")
