package dto

// TaskProgressDTO 任务进度DTO
type TaskProgressDTO struct {
	// 任务ID
	TaskID string `json:"taskId"`
	// 新进度
	Progress int `json:"progress"`
	// 已处理文件数
	ProcessedFiles int `json:"processedFiles"`
}

// TaskResultDTO 任务结果DTO
type TaskResultDTO struct {
	// 任务ID
	TaskID string `json:"taskId"`
	// 执行状态
	Status string `json:"status"`
	// 结果数据
	Result map[string]interface{} `json:"result,omitempty"`
	// 错误信息
	ErrorMessage string `json:"errorMessage,omitempty"`
}
