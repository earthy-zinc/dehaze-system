package model

import "time"

// TaskStatus 任务状态枚举（与前端 SDK 约定一致，使用大写）
type TaskStatus string

const (
	TaskStatusPending    TaskStatus = "PENDING"    // 待执行
	TaskStatusProcessing TaskStatus = "PROCESSING" // 执行中
	TaskStatusCompleted  TaskStatus = "COMPLETED"  // 已完成
	TaskStatusFailed     TaskStatus = "FAILED"     // 失败
	TaskStatusCancelled  TaskStatus = "CANCELLED"  // 已取消
)

// TaskType 任务类型枚举（与前端 SDK 约定一致，使用小写带下划线）
type TaskType string

const (
	TaskTypeUserExport      TaskType = "user_export"
	TaskTypeRoleExport      TaskType = "role_export"
	TaskTypeDeptExport      TaskType = "dept_export"
	TaskTypeMenuExport      TaskType = "menu_export"
	TaskTypeDictExport      TaskType = "dict_export"
	TaskTypeDatasetExport   TaskType = "dataset_export"
	TaskTypeAlgorithmExport TaskType = "algorithm_export"

	TaskTypeUserImport      TaskType = "user_import"
	TaskTypeRoleImport      TaskType = "role_import"
	TaskTypeDeptImport      TaskType = "dept_import"
	TaskTypeMenuImport      TaskType = "menu_import"
	TaskTypeDictImport      TaskType = "dict_import"
	TaskTypeAlgorithmImport TaskType = "algorithm_import"
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
	StartedAt      *time.Time `gorm:"column:started_at;type:datetime;comment:开始时间" json:"startedAt"`
	CompletedAt    *time.Time `gorm:"column:completed_at;type:datetime;comment:完成时间" json:"completedAt"`
	ExpiresAt      *time.Time `gorm:"column:expires_at;type:datetime;index;comment:过期时间" json:"expiresAt"`
	// 客户端幂等键（HTTP Idempotency-Key 头），相同键返回已有任务
	IdempotencyKey *string `gorm:"column:idempotency_key;type:varchar(64);uniqueIndex:idx_idempotency_key;comment:客户端幂等键" json:"idempotencyKey"`
	// MQ 重试次数
	RetryCount int `gorm:"column:retry_count;type:int;not null;default:0;comment:MQ重试次数" json:"retryCount"`
	// 执行 Worker 标识
	WorkerID *string `gorm:"column:worker_id;type:varchar(64);comment:执行Worker标识" json:"workerId"`
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
