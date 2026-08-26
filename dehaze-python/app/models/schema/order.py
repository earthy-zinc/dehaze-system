from pydantic import BaseModel, Field


class OrderCreateForm(BaseModel):
    packageId: int = Field(..., description="套餐ID")
    couponId: int | None = Field(default=None, description="用户优惠券实例ID")
    payMethod: str = Field(..., description="支付方式")
    balanceAmount: int | None = Field(
        default=None, ge=0, description="组合支付时余额部分金额(分)"
    )


class PayRequest(BaseModel):
    payMethod: str = Field(..., description="支付方式")
    balanceAmount: int | None = Field(
        default=None, ge=0, description="组合支付时余额部分金额(分)"
    )


class RefundApplyForm(BaseModel):
    reasonType: str = Field(..., description="售后原因类型(after_sale/force_majeure/merchant/other)")
    reason: str | None = Field(default=None, description="退款原因说明")
    customReason: str | None = Field(default=None, description="自定义补充说明")


class RefundAuditForm(BaseModel):
    approved: bool = Field(..., description="是否通过")
    remark: str = Field(..., description="审核备注")


class AutoRenewConfigForm(BaseModel):
    packageId: int = Field(..., description="套餐ID")
    payMethod: str = Field(..., description="支付方式")
    enabled: bool = Field(..., description="是否启用")


class BalanceRefundForm(BaseModel):
    orderId: int | None = Field(default=None, description="关联订单ID(可空)")
    amount: int | None = Field(default=None, ge=0, description="退款金额(分,为空时按可用余额)")
