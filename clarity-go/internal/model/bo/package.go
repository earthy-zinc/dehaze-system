package bo

// BenefitOverrides 套餐权益覆盖项。字段集须与 python `BENEFIT_FIELDS`（`service/package_service.py`）
// 及 java `model.form.BenefitOverrides` 完全一致——CRUD 保存走 json.Marshal/Unmarshal 该结构，
// 少一个字段就意味着该 key 在保存时被静默丢弃（不报错），而履约侧仍会按 key 读取它。
// 无 binding 约束：python 该模型是裸 `int | None`，不得"顺手"加范围校验。
type BenefitOverrides struct {
	MonthlyDehazeQuota          *int `json:"monthlyDehazeQuota"`
	MonthlyDerainQuota          *int `json:"monthlyDerainQuota"`
	MonthlyDesnowQuota          *int `json:"monthlyDesnowQuota"`
	MonthlyLowlightQuota        *int `json:"monthlyLowlightQuota"`
	MonthlySuperResolutionQuota *int `json:"monthlySuperResolutionQuota"`
	MonthlyDenoiseQuota         *int `json:"monthlyDenoiseQuota"`
	MonthlyInpaintQuota         *int `json:"monthlyInpaintQuota"`
	MonthlyEvaluateQuota        *int `json:"monthlyEvaluateQuota"`
	AiCreditsDaily              *int `json:"aiCreditsDaily"`
	AiCreditsMonthly            *int `json:"aiCreditsMonthly"`
	HistoryRetention            *int `json:"historyRetention"`
	BatchLimit                  *int `json:"batchLimit"`
	Priority                    *int `json:"priority"`
	AdvancedParams              *int `json:"advancedParams"`
	HdExport                    *int `json:"hdExport"`
	ReportExport                *int `json:"reportExport"`
	BatchDownload               *int `json:"batchDownload"`
}

type PackageForm struct {
	ID            int64  `json:"id"`
	Name          string `json:"name" binding:"required,min=2,max=32"`
	PackageType   string `json:"packageType"`
	CreditAmount  *int64 `json:"creditAmount"`
	LevelCode     string `json:"levelCode"`
	Period        string `json:"period"`
	PeriodDays    *int   `json:"periodDays" binding:"omitempty,min=1,max=365"`
	OriginalPrice int64  `json:"originalPrice" binding:"required,min=1"`
	SalePrice     int64  `json:"salePrice" binding:"required,min=1"`
	Description   string `json:"description" binding:"omitempty,max=256"`
	// vip/credit 差异字段的条件必填（levelCode/period/periodDays、creditAmount）由服务层 validatePackageForm 校验
	BenefitOverrides *BenefitOverrides `json:"benefitOverrides"`
	Sort             *int              `json:"sort" binding:"omitempty,min=0,max=999"`
	Status           *int              `json:"status" binding:"omitempty,oneof=0 1"`
}

type PackageStatusForm struct {
	Status int `json:"status"`
}

type CouponForm struct {
	ID           int64   `json:"id"`
	Name         string  `json:"name" binding:"required"`
	Type         string  `json:"type" binding:"required,oneof=full_reduction discount no_threshold trial"`
	FaceValue    int64   `json:"faceValue" binding:"gte=0"`
	Threshold    *int64  `json:"threshold" binding:"omitempty,gte=0"`
	ValidType    string  `json:"validType" binding:"required,oneof=fixed relative"`
	ValidStart   *string `json:"validStart"`
	ValidEnd     *string `json:"validEnd"`
	ValidDays    *int    `json:"validDays" binding:"omitempty,gte=1"`
	TotalQty     int     `json:"totalQty" binding:"gte=0"`
	PerUserLimit int     `json:"perUserLimit" binding:"gte=1"`
	// 适用商品：商品 ID 或商品类型 vip/credit（python list[int|str]），NULL 全部适用
	ApplicableScope []interface{} `json:"applicableScope"`
	Status          *int          `json:"status" binding:"omitempty,oneof=0 1"`
}

type CouponBatchDistributeForm struct {
	CouponID    int64    `json:"couponId"`
	TargetScope string   `json:"targetScope"`
	LevelCodes  []string `json:"levelCodes"`
	UserIDs     []int64  `json:"userIds"`
}
