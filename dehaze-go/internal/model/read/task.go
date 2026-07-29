package read

import "time"

// Task 任务读模型
type Task struct {
	TaskID         string     `json:"taskId"`
	TaskType       string     `json:"taskType"`
	Status         int8       `json:"status"`
	Progress       int        `json:"progress"`
	TotalFiles     int        `json:"totalFiles"`
	ProcessedFiles int        `json:"processedFiles"`
	DownloadURL    string     `json:"downloadUrl"`
	ExpiresAt      *time.Time `json:"expiresAt"`
	CreatedAt      time.Time  `json:"createdAt"`
	StartedAt      *time.Time `json:"startedAt"`
	CompletedAt    *time.Time `json:"completedAt"`
	Error          string     `json:"error"`
}
