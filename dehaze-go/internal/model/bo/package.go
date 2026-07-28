package bo

type BenefitOverrides struct {
	MonthlyDehazeQuota   *int `json:"monthlyDehazeQuota"`
	MonthlyEvaluateQuota *int `json:"monthlyEvaluateQuota"`
	HistoryRetention     *int `json:"historyRetention"`
	BatchLimit           *int `json:"batchLimit"`
	Priority             *int `json:"priority"`
	AdvancedParams       *int `json:"advancedParams"`
	HdExport             *int `json:"hdExport"`
	ReportExport         *int `json:"reportExport"`
	BatchDownload        *int `json:"batchDownload"`
}

type PackageForm struct {
	ID               int64            `json:"id"`
	Name             string           `json:"name"`
	LevelCode        string           `json:"levelCode"`
	Period           string           `json:"period"`
	PeriodDays       int              `json:"periodDays"`
	OriginalPrice    int64            `json:"originalPrice"`
	SalePrice        int64            `json:"salePrice"`
	Description      string           `json:"description"`
	BenefitOverrides *BenefitOverrides `json:"benefitOverrides"`
	Sort             *int             `json:"sort"`
	Status           *int             `json:"status"`
}

type PackageStatusForm struct {
	Status int `json:"status"`
}

type CouponForm struct {
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
	PerUserLimit    int     `json:"perUserLimit"`
	ApplicableScope []int64 `json:"applicableScope"`
	Status          *int    `json:"status"`
}

type CouponBatchDistributeForm struct {
	CouponID    int64    `json:"couponId"`
	TargetScope string   `json:"targetScope"`
	LevelCodes  []string `json:"levelCodes"`
	UserIDs     []int64  `json:"userIds"`
}
