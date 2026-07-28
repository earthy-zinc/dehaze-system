package vo

type CouponVO struct {
	ID              int64   `json:"id"`
	Name            string  `json:"name"`
	Type            string  `json:"type"`
	FaceValue       int64   `json:"faceValue"`
	Threshold       *int64  `json:"threshold"`
	ValidType       string  `json:"validType"`
	ValidStart      *string `json:"validStart"`
	ValidEnd        *string `json:"validEnd"`
	ValidDays       *int    `json:"validDays"`
	TotalQty        int     `json:"totalQty"`
	IssuedQty       int     `json:"issuedQty"`
	UsedQty         int     `json:"usedQty"`
	PerUserLimit    int     `json:"perUserLimit"`
	ApplicableScope []int64 `json:"applicableScope"`
	Status          int     `json:"status"`
	CreateTime      string  `json:"createTime"`
}

type UserCouponVO struct {
	ID              int64   `json:"id"`
	CouponID        int64   `json:"couponId"`
	CouponName      string  `json:"couponName"`
	Type            string  `json:"type"`
	FaceValue       int64   `json:"faceValue"`
	Threshold       *int64  `json:"threshold"`
	Status          int     `json:"status"`
	ReceiveTime     string  `json:"receiveTime"`
	ExpireTime      *string `json:"expireTime"`
	UsedTime        *string `json:"usedTime"`
	UsedOrderID     *int64  `json:"usedOrderId"`
	ApplicableScope []int64 `json:"applicableScope"`
}

type CouponReceiveResult struct {
	UserCouponID int64 `json:"userCouponId"`
}

type CouponBatchDistributeResult struct {
	SuccessCount int `json:"successCount"`
	FailCount    int `json:"failCount"`
}

type CouponCreateResult struct {
	ID int64 `json:"id"`
}
