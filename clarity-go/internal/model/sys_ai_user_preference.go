package model

import "time"

// SysAiUserPreference 用户 AI 偏好设置（每用户一行；default_model 为空表示未设置，按平台默认模型解析）
type SysAiUserPreference struct {
	ID           int64      `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	UserID       int64      `gorm:"column:user_id" json:"userId"`
	DefaultModel *string    `gorm:"column:default_model" json:"defaultModel"`
	Deleted      int64      `gorm:"column:deleted" json:"deleted"`
	CreateBy     *int64     `gorm:"column:create_by" json:"createBy"`
	UpdateBy     *int64     `gorm:"column:update_by" json:"updateBy"`
	CreateTime   time.Time  `gorm:"column:create_time;autoCreateTime" json:"createTime"`
	UpdateTime   *time.Time `gorm:"column:update_time;autoUpdateTime" json:"updateTime"`
}

func (SysAiUserPreference) TableName() string { return "sys_ai_user_preference" }
