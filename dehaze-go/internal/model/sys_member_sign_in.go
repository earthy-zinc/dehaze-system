package model

import "time"

type SysMemberSignIn struct {
	ID             int64     `gorm:"primaryKey;autoIncrement;column:id" json:"id"`
	UserID         int64     `gorm:"column:user_id;type:bigint;not null;uniqueIndex:uk_user_sign_date,priority:1;comment:用户ID" json:"userId"`
	SignDate       time.Time `gorm:"column:sign_date;type:date;not null;uniqueIndex:uk_user_sign_date,priority:2;index:idx_sign_date;comment:签到日期" json:"signDate"`
	ContinuousDays int       `gorm:"column:continuous_days;type:int;not null;default:1;comment:连续签到天数" json:"continuousDays"`
	GrowthValue    int       `gorm:"column:growth_value;type:int;not null;default:0;comment:本次获得成长值" json:"growthValue"`
	CreateTime     time.Time `gorm:"column:create_time;type:datetime;not null;default:CURRENT_TIMESTAMP;comment:创建时间" json:"createTime"`
}

func (SysMemberSignIn) TableName() string {
	return "sys_member_sign_in"
}
