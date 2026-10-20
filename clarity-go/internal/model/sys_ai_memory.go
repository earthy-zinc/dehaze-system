package model

import "time"

// SysAiMemory AI 长期记忆
type SysAiMemory struct {
	ID             int64      `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	UserID         int64      `gorm:"column:user_id" json:"userId"`
	MemoryType     string     `gorm:"column:memory_type" json:"memoryType"`
	Content        string     `gorm:"column:content" json:"content"`
	Metadata       string     `gorm:"column:metadata;type:json;default:null" json:"metadata"`
	Importance     int        `gorm:"column:importance" json:"importance"`
	AccessCount    int        `gorm:"column:access_count" json:"accessCount"`
	LastAccessedAt *time.Time `gorm:"column:last_accessed_at" json:"lastAccessedAt"`
	Source         string     `gorm:"column:source" json:"source"`
	Status         int        `gorm:"column:status" json:"status"`
	Archived       int        `gorm:"column:archived" json:"archived"`
	Deleted        int64      `gorm:"column:deleted" json:"deleted"`
	DeleteTime     *time.Time `gorm:"column:delete_time" json:"deleteTime"`
	CreateBy       *int64     `gorm:"column:create_by" json:"createBy"`
	UpdateBy       *int64     `gorm:"column:update_by" json:"updateBy"`
	CreateTime     time.Time  `gorm:"column:create_time;autoCreateTime" json:"createTime"`
	UpdateTime     *time.Time `gorm:"column:update_time;autoUpdateTime" json:"updateTime"`
}

func (SysAiMemory) TableName() string { return "sys_ai_memory" }
