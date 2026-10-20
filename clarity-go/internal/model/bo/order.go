package bo

type OrderCreateForm struct {
	PackageID     int64  `json:"packageId"`
	CouponID      *int64 `json:"couponId"`
	PayMethod     string `json:"payMethod"`
	BalanceAmount *int64 `json:"balanceAmount"`
}

type PayRequest struct {
	PayMethod     string `json:"payMethod"`
	Channel       string `json:"channel"`
	BalanceAmount *int64 `json:"balanceAmount"`
}

type RefundApplyForm struct {
	ReasonType   string `json:"reasonType" binding:"required,oneof=after_sale force_majeure merchant other"`
	CustomReason string `json:"customReason"`
}

type RefundAuditForm struct {
	Approved bool   `json:"approved"`
	Remark   string `json:"remark"`
}

type BalanceRefundForm struct {
	OrderID *int64 `json:"orderId"`
	Amount  *int64 `json:"amount" binding:"omitempty,gte=0"`
}

type BalanceRefundAuditForm struct {
	Remark  string `json:"remark"`
	Channel string `json:"channel" binding:"omitempty,oneof=wechat alipay"`
}

type RechargeCreateForm struct {
	Amount    int64  `json:"amount" binding:"required,gt=0"`
	PayMethod string `json:"payMethod" binding:"required,oneof=wechat alipay"`
}

type AutoRenewConfigForm struct {
	PackageID int64  `json:"packageId"`
	PayMethod string `json:"payMethod"`
	Enabled   bool   `json:"enabled"`
}
