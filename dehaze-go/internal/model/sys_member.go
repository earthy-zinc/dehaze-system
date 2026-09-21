package model

import "time"

type SysMember struct {
	BaseModel
	UserID                      int64      `gorm:"column:user_id;type:bigint;not null;uniqueIndex:uk_user_id;comment:用户ID" json:"userId"`
	LevelCode                   string     `gorm:"column:level_code;type:varchar(16);not null;default:level_0;index:idx_level_code;comment:会员等级" json:"levelCode"`
	LevelSource                 string     `gorm:"column:level_source;type:varchar(16);not null;default:growth;comment:等级来源" json:"levelSource"`
	GrowthValue                 int64      `gorm:"column:growth_value;type:bigint;not null;default:0;comment:成长值" json:"growthValue"`
	TotalConsumption            int64      `gorm:"column:total_consumption;type:bigint;not null;default:0;comment:累计消费金额（分）" json:"totalConsumption"`
	ExpireTime                  *time.Time `gorm:"column:expire_time;type:datetime;index:idx_expire_time;comment:套餐到期时间" json:"expireTime"`
	BecomeMemberTime            *time.Time `gorm:"column:become_member_time;type:datetime;comment:首次成为会员时间" json:"becomeMemberTime"`
	MonthlyDehazeQuota          int        `gorm:"column:monthly_dehaze_quota;type:int;not null;default:0;comment:本月去雾配额" json:"monthlyDehazeQuota"`
	MonthlyDehazeUsed           int        `gorm:"column:monthly_dehaze_used;type:int;not null;default:0;comment:本月已用去雾次数" json:"monthlyDehazeUsed"`
	MonthlyDerainQuota          int        `gorm:"column:monthly_derain_quota;type:int;not null;default:0;comment:本月去雨配额" json:"monthlyDerainQuota"`
	MonthlyDerainUsed           int        `gorm:"column:monthly_derain_used;type:int;not null;default:0;comment:本月已用去雨次数" json:"monthlyDerainUsed"`
	MonthlyDesnowQuota          int        `gorm:"column:monthly_desnow_quota;type:int;not null;default:0;comment:本月去雪配额" json:"monthlyDesnowQuota"`
	MonthlyDesnowUsed           int        `gorm:"column:monthly_desnow_used;type:int;not null;default:0;comment:本月已用去雪次数" json:"monthlyDesnowUsed"`
	MonthlyLowlightQuota        int        `gorm:"column:monthly_lowlight_quota;type:int;not null;default:0;comment:本月低光增强配额" json:"monthlyLowlightQuota"`
	MonthlyLowlightUsed         int        `gorm:"column:monthly_lowlight_used;type:int;not null;default:0;comment:本月已用低光增强次数" json:"monthlyLowlightUsed"`
	MonthlySuperResolutionQuota int        `gorm:"column:monthly_super_resolution_quota;type:int;not null;default:0;comment:本月超分辨率配额" json:"monthlySuperResolutionQuota"`
	MonthlySuperResolutionUsed  int        `gorm:"column:monthly_super_resolution_used;type:int;not null;default:0;comment:本月已用超分辨率次数" json:"monthlySuperResolutionUsed"`
	MonthlyDenoiseQuota         int        `gorm:"column:monthly_denoise_quota;type:int;not null;default:0;comment:本月去噪配额" json:"monthlyDenoiseQuota"`
	MonthlyDenoiseUsed          int        `gorm:"column:monthly_denoise_used;type:int;not null;default:0;comment:本月已用去噪次数" json:"monthlyDenoiseUsed"`
	MonthlyInpaintQuota         int        `gorm:"column:monthly_inpaint_quota;type:int;not null;default:0;comment:本月图像修复配额" json:"monthlyInpaintQuota"`
	MonthlyInpaintUsed          int        `gorm:"column:monthly_inpaint_used;type:int;not null;default:0;comment:本月已用图像修复次数" json:"monthlyInpaintUsed"`
	MonthlyEvaluateQuota        int        `gorm:"column:monthly_evaluate_quota;type:int;not null;default:0;comment:本月评估配额" json:"monthlyEvaluateQuota"`
	MonthlyEvaluateUsed         int        `gorm:"column:monthly_evaluate_used;type:int;not null;default:0;comment:本月已用评估次数" json:"monthlyEvaluateUsed"`
	QuotaResetMonth             *int       `gorm:"column:quota_reset_month;type:int;index:idx_quota_reset_month;comment:配额所属月份" json:"quotaResetMonth"`
	Status                      int8       `gorm:"column:status;type:tinyint;not null;default:1;index:idx_status;comment:状态(1:正常;0:冻结)" json:"status"`
	FrozenReason                string     `gorm:"column:frozen_reason;type:varchar(256);comment:冻结原因" json:"frozenReason"`
	FrozenTime                  *time.Time `gorm:"column:frozen_time;type:datetime;comment:冻结时间" json:"frozenTime"`
	Deleted                     int64      `gorm:"column:deleted;type:bigint;not null;default:0;comment:逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)" json:"deleted"`
}

func (SysMember) TableName() string {
	return "sys_member"
}
