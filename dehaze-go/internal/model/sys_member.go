package model

import "time"

type SysMember struct {
	BaseModel
	UserID                int64      `gorm:"column:user_id;type:bigint;not null;uniqueIndex:uk_user_id;comment:用户ID" json:"userId"`
	LevelCode             string     `gorm:"column:level_code;type:varchar(16);not null;default:level_0;index:idx_level_code;comment:会员等级" json:"levelCode"`
	LevelSource           string     `gorm:"column:level_source;type:varchar(16);not null;default:growth;comment:等级来源" json:"levelSource"`
	GrowthValue           int64      `gorm:"column:growth_value;type:bigint;not null;default:0;comment:成长值" json:"growthValue"`
	TotalConsumption      int64      `gorm:"column:total_consumption;type:bigint;not null;default:0;comment:累计消费金额（分）" json:"totalConsumption"`
	ExpireTime            *time.Time `gorm:"column:expire_time;type:datetime;index:idx_expire_time;comment:套餐到期时间" json:"expireTime"`
	BecomeMemberTime      *time.Time `gorm:"column:become_member_time;type:datetime;comment:首次成为会员时间" json:"becomeMemberTime"`
	MonthlyDehazeQuota    int        `gorm:"column:monthly_dehaze_quota;type:int;not null;default:0;comment:本月去雾配额" json:"monthlyDehazeQuota"`
	MonthlyDehazeUsed     int        `gorm:"column:monthly_dehaze_used;type:int;not null;default:0;comment:本月已用去雾次数" json:"monthlyDehazeUsed"`
	MonthlyEvaluateQuota  int        `gorm:"column:monthly_evaluate_quota;type:int;not null;default:0;comment:本月评估配额" json:"monthlyEvaluateQuota"`
	MonthlyEvaluateUsed   int        `gorm:"column:monthly_evaluate_used;type:int;not null;default:0;comment:本月已用评估次数" json:"monthlyEvaluateUsed"`
	QuotaResetMonth       *int       `gorm:"column:quota_reset_month;type:int;index:idx_quota_reset_month;comment:配额所属月份" json:"quotaResetMonth"`
	Status                int8       `gorm:"column:status;type:tinyint;not null;default:1;index:idx_status;comment:状态(1:正常;0:冻结)" json:"status"`
	FrozenReason          string     `gorm:"column:frozen_reason;type:varchar(256);comment:冻结原因" json:"frozenReason"`
	FrozenTime            *time.Time `gorm:"column:frozen_time;type:datetime;comment:冻结时间" json:"frozenTime"`
	Deleted               int8       `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysMember) TableName() string {
	return "sys_member"
}
