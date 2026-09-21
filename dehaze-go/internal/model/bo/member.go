package bo

type MemberLevelAdjustForm struct {
	LevelCode  string  `json:"levelCode"`
	ExpireTime *string `json:"expireTime"`
	Reason     string  `json:"reason"`
}

type MemberGrowthAdjustForm struct {
	ChangeValue int `json:"changeValue"`
	// 调整原因上限 256，与 DB varchar(256) 及 Python 端 max_length 对齐，防超长触发 500
	Reason string `json:"reason" binding:"max=256"`
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
	MaxDevices           *int    `json:"maxDevices" binding:"omitempty,min=1"`
	Priority             *int    `json:"priority" binding:"omitempty,min=1,max=4"`
	AdvancedParams       *int    `json:"advancedParams" binding:"omitempty,oneof=0 1"`
	HdExport             *int    `json:"hdExport" binding:"omitempty,oneof=0 1"`
	ReportExport         *int    `json:"reportExport" binding:"omitempty,oneof=0 1"`
	BatchDownload        *int    `json:"batchDownload" binding:"omitempty,oneof=0 1"`
	Sort                 *int    `json:"sort" binding:"omitempty,min=0"`
	Status               *int    `json:"status" binding:"omitempty,oneof=0 1"`
}
