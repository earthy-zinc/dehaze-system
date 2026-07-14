package vo

import "time"

// TaskVO 任务视图对象
type TaskVO struct {
	// 任务唯一ID
	TaskID string `json:"taskId"`
	// 任务状态：pending, processing, completed, failed, cancelled
	Status string `json:"status"`
	// 进度（0-100）
	Progress int `json:"progress"`
	// 文件总数
	TotalFiles int `json:"totalFiles"`
	// 已处理文件数
	ProcessedFiles int `json:"processedFiles"`
	// 下载链接
	DownloadURL string `json:"downloadUrl"`
	// 过期时间
	ExpiresAt *time.Time `json:"expiresAt"`
	// 创建时间
	CreatedAt time.Time `json:"createdAt"`
	// 开始时间
	StartedAt *time.Time `json:"startedAt"`
	// 完成时间
	CompletedAt *time.Time `json:"completedAt"`
	// 错误信息
	Error string `json:"error"`
	// 客户端幂等键
	IdempotencyKey *string `json:"idempotencyKey"`
	// MQ 重试次数
	RetryCount int `json:"retryCount"`
	// 执行 Worker 标识
	WorkerID *string `json:"workerId"`
}

// TaskDetailVO 任务详情视图对象
type TaskDetailVO struct {
	TaskVO
	// 任务名称
	Name string `json:"name"`
	// 任务类型
	Type string `json:"type"`
	// 算法ID
	AlgorithmID int64 `json:"algorithmId"`
	// 算法名称
	AlgorithmName string `json:"algorithmName"`
	// 数据集ID
	DatasetID int64 `json:"datasetId"`
	// 数据集名称
	DatasetName string `json:"datasetName"`
	// 任务参数
	Parameters string `json:"parameters"`
	// 结果列表
	Results []TaskResultItemVO `json:"results,omitempty"`
}

// TaskResultItemVO 任务结果项视图对象
type TaskResultItemVO struct {
	// 数据项ID
	ItemID int64 `json:"itemId"`
	// 数据项名称
	ItemName string `json:"itemName"`
	// 状态
	Status string `json:"status"`
	// 输入文件路径
	InputPath string `json:"inputPath"`
	// 输出文件路径
	OutputPath string `json:"outputPath,omitempty"`
	// 错误信息
	Error string `json:"error,omitempty"`
}

// BatchOperationResultVO 批量操作结果VO
type BatchOperationResultVO struct {
	// 总数
	Total int `json:"total"`
	// 成功数
	Succeeded int `json:"succeeded"`
	// 失败数
	Failed int `json:"failed"`
	// 详细结果列表
	Results []BatchOperationItemVO `json:"results"`
}

// BatchOperationItemVO 批量操作单项结果VO
type BatchOperationItemVO struct {
	// ID
	ID int64 `json:"id"`
	// 状态：success, failed
	Status string `json:"status"`
	// 消息
	Message string `json:"message"`
	// 错误码
	ErrorCode string `json:"errorCode,omitempty"`
}

// BatchUploadResultVO 批量上传结果VO
type BatchUploadResultVO struct {
	// 总数
	Total int `json:"total"`
	// 成功数
	Succeeded int `json:"succeeded"`
	// 失败数
	Failed int `json:"failed"`
	// 成功项目列表
	SuccessItems []BatchUploadSuccessItemVO `json:"successItems"`
	// 失败项目列表
	FailedItems []BatchUploadFailedItemVO `json:"failedItems"`
}

// BatchUploadSuccessItemVO 批量上传成功项VO
type BatchUploadSuccessItemVO struct {
	// 数据项ID
	ItemID int64 `json:"itemId"`
	// 数据项名称
	ItemName string `json:"itemName"`
	// 上传的文件ID列表
	FileIDs []int64 `json:"fileIds"`
}

// BatchUploadFailedItemVO 批量上传失败项VO
type BatchUploadFailedItemVO struct {
	// 原始文件名
	FileName string `json:"fileName"`
	// 错误消息
	Error string `json:"error"`
}
