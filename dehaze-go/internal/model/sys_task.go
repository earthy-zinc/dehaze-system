package model

import "time"

// TaskStatus 任务状态枚举
type TaskStatus string

const (
	TaskStatusPending    TaskStatus = "pending"    // 待执行
	TaskStatusProcessing TaskStatus = "processing" // 执行中
	TaskStatusCompleted  TaskStatus = "completed"  // 已完成
	TaskStatusFailed     TaskStatus = "failed"     // 失败
	TaskStatusCancelled  TaskStatus = "cancelled"  // 已取消
)

// TaskType 任务类型枚举
type TaskType string

const (
	TaskTypeExport      TaskType = "export"      // 导出任务
	TaskTypeImport      TaskType = "import"      // 导入任务
	TaskTypeThumbnail   TaskType = "thumbnail"   // 缩略图生成任务
	TaskTypeCompression TaskType = "compression" // 压缩任务
	TaskTypeCleanup     TaskType = "cleanup"     // 清理任务
)

// SysTask 系统任务表
type SysTask struct {
	BaseModel
	TaskID         string     `gorm:"column:task_id;type:varchar(64);uniqueIndex;not null;comment:任务唯一ID" json:"taskId"`
	TaskType       TaskType   `gorm:"column:task_type;type:varchar(32);not null;comment:任务类型" json:"taskType"`
	Status         TaskStatus `gorm:"column:status;type:varchar(32);not null;comment:任务状态" json:"status"`
	Progress       int        `gorm:"column:progress;type:int;default:0;comment:进度(0-100)" json:"progress"`
	TotalFiles     int        `gorm:"column:total_files;type:int;default:0;comment:文件总数" json:"totalFiles"`
	ProcessedFiles int        `gorm:"column:processed_files;type:int;default:0;comment:已处理文件数" json:"processedFiles"`
	Params         string     `gorm:"column:params;type:text;comment:任务参数JSON" json:"params"`
	Result         string     `gorm:"column:result;type:text;comment:执行结果JSON" json:"result"`
	ErrorMessage   string     `gorm:"column:error_message;type:varchar(1024);comment:错误信息" json:"errorMessage"`
	CreatedBy      int64      `gorm:"column:created_by;type:bigint;comment:创建人ID" json:"createdBy"`
	StartedAt      *time.Time `gorm:"column:started_at;type:datetime;comment:开始时间" json:"startedAt"`
	CompletedAt    *time.Time `gorm:"column:completed_at;type:datetime;comment:完成时间" json:"completedAt"`
	ExpiresAt      *time.Time `gorm:"column:expires_at;type:datetime;index;comment:过期时间" json:"expiresAt"`
}

// TableName 指定表名
func (SysTask) TableName() string {
	return "sys_task"
}

// IsExpired 检查任务是否已过期
func (t *SysTask) IsExpired() bool {
	if t.ExpiresAt == nil {
		return false
	}
	return time.Now().After(*t.ExpiresAt)
}

// IsCompleted 检查任务是否已完成（成功或失败）
func (t *SysTask) IsCompleted() bool {
	return t.Status == TaskStatusCompleted || t.Status == TaskStatusFailed || t.Status == TaskStatusCancelled
}

// CanCancel 检查任务是否可以取消
func (t *SysTask) CanCancel() bool {
	return t.Status == TaskStatusPending || t.Status == TaskStatusProcessing
}
