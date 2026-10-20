package query

// TaskPageQuery 任务分页查询对象
type TaskPageQuery struct {
	// 任务状态（可选）
	Status *int8 `json:"status"`
	// 任务类型（可选）
	TaskType string `json:"taskType"`
	// 任务类别（可选）：import / export
	TaskCategory string `json:"taskCategory"`
	// 创建者 ID（可选，为 0 时查询全部）
	UserID int64 `json:"userId"`
	// 页码
	PageNum int `json:"pageNum"`
	// 每页条数
	PageSize int `json:"pageSize"`
}
