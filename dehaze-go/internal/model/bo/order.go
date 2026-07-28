package bo

type OrderCreateForm struct {
	PackageID int64  `json:"packageId"`
	CouponID  *int64 `json:"couponId"`
	PayMethod string `json:"payMethod"`
}

type PayRequest struct {
	PayMethod string `json:"payMethod"`
}

type RefundApplyForm struct {
	Reason        string `json:"reason"`
	CustomReason  string `json:"customReason"`
}

type RefundAuditForm struct {
	Approved bool   `json:"approved"`
	Remark   string `json:"remark"`
}

type AutoRenewConfigForm struct {
	PackageID int64  `json:"packageId"`
	PayMethod string `json:"payMethod"`
	Enabled   bool   `json:"enabled"`
}
