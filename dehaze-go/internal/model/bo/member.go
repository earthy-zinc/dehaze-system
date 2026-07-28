package bo

type MemberLevelAdjustForm struct {
	LevelCode  string  `json:"levelCode"`
	ExpireTime *string `json:"expireTime"`
	Reason     string  `json:"reason"`
}

type MemberGrowthAdjustForm struct {
	ChangeValue int    `json:"changeValue"`
	Reason      string `json:"reason"`
}

type MemberStatusForm struct {
	Status int    `json:"status"`
	Reason string `json:"reason"`
}

type BenefitForm struct {
	LevelName            *string `json:"levelName"`
	GrowthMin            *int64  `json:"growthMin"`
	GrowthMax            *int64  `json:"growthMax"`
	MonthlyDehazeQuota   *int    `json:"monthlyDehazeQuota"`
	MonthlyEvaluateQuota *int    `json:"monthlyEvaluateQuota"`
	HistoryRetention     *int    `json:"historyRetention"`
	BatchLimit           *int    `json:"batchLimit"`
	Priority             *int    `json:"priority"`
	AdvancedParams       *int    `json:"advancedParams"`
	HdExport             *int    `json:"hdExport"`
	ReportExport         *int    `json:"reportExport"`
	BatchDownload        *int    `json:"batchDownload"`
	Sort                 *int    `json:"sort"`
	Status               *int    `json:"status"`
}
