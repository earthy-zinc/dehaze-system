package query

// TaskPageQuery 任务分页查询对象
type TaskPageQuery struct {
	// 任务状态（可选）
	Status string `json:"status"`
	// 任务类型（可选）
	TaskType string `json:"taskType"`
	// 创建者 ID（可选，为 0 时查询全部）
	UserID int64 `json:"userId"`
	// 页码
	PageNum int `json:"pageNum"`
	// 每页条数
	PageSize int `json:"pageSize"`
}
