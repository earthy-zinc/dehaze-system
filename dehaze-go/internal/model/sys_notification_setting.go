package model

import "time"

type SysNotificationSetting struct {
	ID          int64     `gorm:"primaryKey;autoIncrement;column:id" json:"id"`
	UserID      int64     `gorm:"column:user_id;type:bigint;uniqueIndex:uk_user_id;not null;comment:用户ID" json:"userId"`
	PushEnabled int8      `gorm:"column:push_enabled;type:tinyint;not null;default:1;comment:APP推送总开关" json:"pushEnabled"`
	DndEnabled  int8      `gorm:"column:dnd_enabled;type:tinyint;not null;default:0;comment:免打扰开关" json:"dndEnabled"`
	DndStart    string    `gorm:"column:dnd_start;type:time;default:22:00:00;comment:免打扰开始时间" json:"dndStart"`
	DndEnd      string    `gorm:"column:dnd_end;type:time;default:08:00:00;comment:免打扰结束时间" json:"dndEnd"`
	Preferences string    `gorm:"column:preferences;type:json;comment:细粒度偏好JSON" json:"preferences"`
	Deleted     int8      `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识" json:"-"`
	CreateTime  time.Time `gorm:"column:create_time;type:datetime;autoCreateTime;comment:创建时间" json:"createTime"`
	UpdateTime  time.Time `gorm:"column:update_time;type:datetime;autoUpdateTime;comment:更新时间" json:"updateTime"`
	CreateBy    *int64    `gorm:"column:create_by;comment:创建人ID" json:"createBy"`
	UpdateBy    *int64    `gorm:"column:update_by;comment:修改人ID" json:"updateBy"`
}

func (SysNotificationSetting) TableName() string {
	return "sys_notification_setting"
}
