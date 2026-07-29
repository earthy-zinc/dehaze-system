package vo

type MyOrderVO struct {
	ID                int64   `json:"id"`
	OrderNo           string  `json:"orderNo"`
	PackageName       string  `json:"packageName"`
	PackageLevel      string  `json:"packageLevel"`
	PayableAmount     int64   `json:"payableAmount"`
	PaidAmount        int64   `json:"paidAmount"`
	PayMethod         *string `json:"payMethod"`
	Status            string  `json:"status"`
	CreateTime        string  `json:"createTime"`
	PaidTime          *string `json:"paidTime"`
	PackageExpireTime *string `json:"packageExpireTime"`
}

type OrderPageVO struct {
	MyOrderVO
	UserID         int64  `json:"userId"`
	Username       string `json:"username"`
	OriginalPrice  int64  `json:"originalPrice"`
	DiscountAmount int64  `json:"discountAmount"`
	CouponAmount   int64  `json:"couponAmount"`
}

type PaymentRecordVO struct {
	ID            int64   `json:"id"`
	PaymentNo     string  `json:"paymentNo"`
	Channel       string  `json:"channel"`
	Amount        int64   `json:"amount"`
	Status        int     `json:"status"`
	CallbackTime  *string `json:"callbackTime"`
	CreateTime    string  `json:"createTime"`
}

type RefundRecordVO struct {
	ID              int64   `json:"id"`
	RefundNo        string  `json:"refundNo"`
	OrderID         int64   `json:"orderId"`
	OrderNo         string  `json:"orderNo"`
	UserID          int64   `json:"userId"`
	Username        string  `json:"username"`
	RefundAmount    int64   `json:"refundAmount"`
	Reason          string  `json:"reason"`
	UsedQuota       int     `json:"usedQuota"`
	Status          string  `json:"status"`
	Channel         *string `json:"channel"`
	ChannelRefundNo string  `json:"channelRefundNo"`
	ApplyTime       string  `json:"applyTime"`
	AuditTime       *string `json:"auditTime"`
	AuditorID       *int64  `json:"auditorId"`
	AuditRemark     string  `json:"auditRemark"`
	RefundTime      *string `json:"refundTime"`
	ErrorMessage    string  `json:"errorMessage"`
}

type OrderDetailVO struct {
	OrderPageVO
	ExpireTime      string             `json:"expireTime"`
	EffectiveTime   *string            `json:"effectiveTime"`
	CancelReason    string             `json:"cancelReason"`
	IsAutoRenew     int                `json:"isAutoRenew"`
	PaymentRecords  []PaymentRecordVO  `json:"paymentRecords"`
	RefundRecord    *RefundRecordVO    `json:"refundRecord"`
}

type PayResult struct {
	OrderNo   string  `json:"orderNo"`
	PayMethod string  `json:"payMethod"`
	PayURL    string  `json:"payUrl"`
	QRCode    string  `json:"qrCode"`
	Paid      bool    `json:"paid"`
}

type OrderStatsVO struct {
	TotalOrders           int64                      `json:"totalOrders"`
	TotalRevenue          int64                      `json:"totalRevenue"`
	TotalRefund           int64                      `json:"totalRefund"`
	RefundRate            float64                    `json:"refundRate"`
	StatusDistribution    map[string]int64           `json:"statusDistribution"`
	PayMethodDistribution map[string]int64           `json:"payMethodDistribution"`
	PackageDistribution   []OrderPackageStatItem     `json:"packageDistribution"`
	DailyStats            []OrderDailyStatItem       `json:"dailyStats"`
}

type OrderPackageStatItem struct {
	PackageID   int64  `json:"packageId"`
	PackageName string `json:"packageName"`
	Count       int64  `json:"count"`
	Revenue     int64  `json:"revenue"`
}

type OrderDailyStatItem struct {
	Date    string `json:"date"`
	Count   int64  `json:"count"`
	Revenue int64  `json:"revenue"`
}

type AutoRenewConfigVO struct {
	UserID        int64   `json:"userId"`
	PackageID     int64   `json:"packageId"`
	PackageName   string  `json:"packageName"`
	PayMethod     string  `json:"payMethod"`
	Enabled       bool    `json:"enabled"`
	NextRenewTime *string `json:"nextRenewTime"`
	FailCount     int     `json:"failCount"`
	CloseReason   string  `json:"closeReason"`
}
