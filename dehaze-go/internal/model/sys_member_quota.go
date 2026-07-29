package model

import "time"

type SysMemberQuota struct {
	ID            int64     `gorm:"primaryKey;autoIncrement;column:id" json:"id"`
	UserID        int64     `gorm:"column:user_id;type:bigint;not null;uniqueIndex:uk_user_month,priority:1;comment:用户ID" json:"userId"`
	QuotaMonth    int       `gorm:"column:quota_month;type:int;not null;uniqueIndex:uk_user_month,priority:2;index:idx_quota_month;comment:配额月份" json:"quotaMonth"`
	LevelCode     string    `gorm:"column:level_code;type:varchar(16);not null;comment:当月会员等级" json:"levelCode"`
	DehazeQuota   int       `gorm:"column:dehaze_quota;type:int;not null;default:0;comment:当月去雾配额" json:"dehazeQuota"`
	DehazeUsed    int       `gorm:"column:dehaze_used;type:int;not null;default:0;comment:当月已用去雾次数" json:"dehazeUsed"`
	EvaluateQuota int       `gorm:"column:evaluate_quota;type:int;not null;default:0;comment:当月评估配额" json:"evaluateQuota"`
	EvaluateUsed  int       `gorm:"column:evaluate_used;type:int;not null;default:0;comment:当月已用评估次数" json:"evaluateUsed"`
	ResetTime     time.Time `gorm:"column:reset_time;type:datetime;not null;comment:配额重置时间" json:"resetTime"`
	Deleted       int8      `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识" json:"-"`
	CreateTime    time.Time `gorm:"column:create_time;type:datetime;default:CURRENT_TIMESTAMP;comment:创建时间" json:"createTime"`
	UpdateTime    time.Time `gorm:"column:update_time;type:datetime;default:CURRENT_TIMESTAMP;comment:更新时间" json:"updateTime"`
	CreateBy      *int64    `gorm:"column:create_by;comment:创建人ID" json:"createBy"`
	UpdateBy      *int64    `gorm:"column:update_by;comment:修改人ID" json:"updateBy"`
}

func (SysMemberQuota) TableName() string {
	return "sys_member_quota"
}
