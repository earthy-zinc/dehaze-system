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
	GrowthMin            *int64  `json:"growthMin" binding:"omitempty,min=0"`
	GrowthMax            *int64  `json:"growthMax" binding:"omitempty,min=0"`
	MonthlyDehazeQuota   *int    `json:"monthlyDehazeQuota" binding:"omitempty,min=0"`
	MonthlyEvaluateQuota *int    `json:"monthlyEvaluateQuota" binding:"omitempty,min=0"`
	HistoryRetention     *int    `json:"historyRetention" binding:"omitempty,min=0"`
	BatchLimit           *int    `json:"batchLimit" binding:"omitempty,min=0"`
	Priority             *int    `json:"priority" binding:"omitempty,min=1,max=4"`
	AdvancedParams       *int    `json:"advancedParams" binding:"omitempty,oneof=0 1"`
	HdExport             *int    `json:"hdExport" binding:"omitempty,oneof=0 1"`
	ReportExport         *int    `json:"reportExport" binding:"omitempty,oneof=0 1"`
	BatchDownload        *int    `json:"batchDownload" binding:"omitempty,oneof=0 1"`
	Sort                 *int    `json:"sort" binding:"omitempty,min=0"`
	Status               *int    `json:"status" binding:"omitempty,oneof=0 1"`
}
