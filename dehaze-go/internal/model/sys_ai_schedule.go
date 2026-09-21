package model

import "time"

// SysAiSchedule AI 对话定时任务
type SysAiSchedule struct {
	ID              int64      `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	UserID          int64      `gorm:"column:user_id" json:"userId"`
	Name            string     `gorm:"column:name" json:"name"`
	Cron            string     `gorm:"column:cron" json:"cron"`
	Timezone        string     `gorm:"column:timezone" json:"timezone"`
	Input           string     `gorm:"column:input;type:json;default:null" json:"input"`
	Output          string     `gorm:"column:output;type:json;default:null" json:"output"`
	Enabled         int        `gorm:"column:enabled" json:"enabled"`
	Status          int        `gorm:"column:status" json:"status"`
	CircuitStreak   int        `gorm:"column:circuit_streak" json:"circuitStreak"`
	NextTriggerTime *time.Time `gorm:"column:next_trigger_time" json:"nextTriggerTime"`
	Deleted         int64      `gorm:"column:deleted" json:"deleted"`
	CreateBy        *int64     `gorm:"column:create_by" json:"createBy"`
	UpdateBy        *int64     `gorm:"column:update_by" json:"updateBy"`
	CreateTime      time.Time  `gorm:"column:create_time;autoCreateTime" json:"createTime"`
	UpdateTime      *time.Time `gorm:"column:update_time;autoUpdateTime" json:"updateTime"`
}

func (SysAiSchedule) TableName() string { return "sys_ai_schedule" }

// SysAiScheduleRun 定时任务执行历史（只追加）
type SysAiScheduleRun struct {
	ID             int64     `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	ScheduleID     int64     `gorm:"column:schedule_id" json:"scheduleId"`
	UserID         int64     `gorm:"column:user_id" json:"userId"`
	WindowStart    time.Time `gorm:"column:window_start" json:"windowStart"`
	Status         int       `gorm:"column:status" json:"status"`
	SkipReason     *string   `gorm:"column:skip_reason" json:"skipReason"`
	Credits        *float64  `gorm:"column:credits" json:"credits"`
	DurationMs     *int      `gorm:"column:duration_ms" json:"durationMs"`
	ErrorMsg       *string   `gorm:"column:error_msg" json:"errorMsg"`
	ConversationID *int64    `gorm:"column:conversation_id" json:"conversationId"`
	RequestID      *string   `gorm:"column:request_id" json:"requestId"`
	CreateTime     time.Time `gorm:"column:create_time;autoCreateTime" json:"createTime"`
}

func (SysAiScheduleRun) TableName() string { return "sys_ai_schedule_run" }
