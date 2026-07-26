package bo

// TaskCreateForm 通用任务创建表单
type TaskCreateForm struct {
	// 任务类型（如 user_export/role_import/dataset_export 等）
	TaskType string `json:"taskType" binding:"required"`
	// 任务参数（任意可序列化结构，由具体策略解析）
	Params interface{} `json:"params"`
}

// BatchDeleteForm 批量删除表单
type BatchDeleteForm struct {
	// 要删除的ID列表
	IDs []int64 `json:"ids" binding:"required,min=1"`
	// 是否强制删除（跳过检查）
	Force bool `json:"force"`
}
