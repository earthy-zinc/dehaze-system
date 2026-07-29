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
	ID               int64             `json:"id"`
	Name             string            `json:"name" binding:"required,min=2,max=32"`
	LevelCode        string            `json:"levelCode" binding:"required"`
	Period           string            `json:"period" binding:"required,oneof=monthly quarterly yearly"`
	PeriodDays       int               `json:"periodDays" binding:"required,min=1,max=365"`
	OriginalPrice    int64             `json:"originalPrice" binding:"required,min=1"`
	SalePrice        int64             `json:"salePrice" binding:"required,min=1"`
	Description      string            `json:"description" binding:"omitempty,max=256"`
	BenefitOverrides *BenefitOverrides `json:"benefitOverrides"`
	Sort             *int              `json:"sort" binding:"omitempty,min=0,max=999"`
	Status           *int              `json:"status" binding:"omitempty,oneof=0 1"`
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
