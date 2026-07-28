package model

type SysMemberBenefit struct {
	BaseModel
	LevelCode            string `gorm:"column:level_code;type:varchar(16);not null;uniqueIndex:uk_level_code;comment:会员等级" json:"levelCode"`
	LevelName            string `gorm:"column:level_name;type:varchar(32);not null;comment:等级名称" json:"levelName"`
	GrowthMin            int64  `gorm:"column:growth_min;type:bigint;not null;default:0;comment:成长值下限" json:"growthMin"`
	GrowthMax            int64  `gorm:"column:growth_max;type:bigint;not null;default:0;comment:成长值上限（0表示无上限）" json:"growthMax"`
	MonthlyDehazeQuota   int    `gorm:"column:monthly_dehaze_quota;type:int;not null;default:0;comment:月度去雾次数配额" json:"monthlyDehazeQuota"`
	MonthlyEvaluateQuota int    `gorm:"column:monthly_evaluate_quota;type:int;not null;default:0;comment:月度评估次数配额" json:"monthlyEvaluateQuota"`
	HistoryRetention     int    `gorm:"column:history_retention;type:int;not null;default:0;comment:历史记录保留条数" json:"historyRetention"`
	BatchLimit           int    `gorm:"column:batch_limit;type:int;not null;default:0;comment:批量处理上限" json:"batchLimit"`
	Priority             int8   `gorm:"column:priority;type:tinyint;not null;default:1;comment:处理优先级" json:"priority"`
	AdvancedParams       int8   `gorm:"column:advanced_params;type:tinyint;not null;default:0;comment:高级参数调节" json:"advancedParams"`
	HdExport             int8   `gorm:"column:hd_export;type:tinyint;not null;default:0;comment:高清图导出" json:"hdExport"`
	ReportExport         int8   `gorm:"column:report_export;type:tinyint;not null;default:0;comment:对比报告导出" json:"reportExport"`
	BatchDownload        int8   `gorm:"column:batch_download;type:tinyint;not null;default:0;comment:批量打包下载" json:"batchDownload"`
	Sort                 int    `gorm:"column:sort;type:int;not null;default:0;comment:排序值" json:"sort"`
	Status               int8   `gorm:"column:status;type:tinyint;not null;default:1;comment:状态" json:"status"`
	Deleted              int8   `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysMemberBenefit) TableName() string {
	return "sys_member_benefit"
}
