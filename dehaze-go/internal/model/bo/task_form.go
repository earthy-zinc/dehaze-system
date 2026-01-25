package bo

// TaskBO 任务业务对象（用于创建/更新任务）
type TaskBO struct {
	// 任务ID
	ID int64 `json:"id"`
	// 任务名称
	Name string `json:"name" binding:"required"`
	// 任务类型
	Type string `json:"type" binding:"required"`
	// 算法ID
	AlgorithmID int64 `json:"algorithmId" binding:"required"`
	// 数据集ID
	DatasetID int64 `json:"datasetId" binding:"required"`
	// 数据项ID列表（可选，不传则处理整个数据集）
	ItemIDs []int64 `json:"itemIds"`
	// 任务参数（JSON格式）
	Parameters string `json:"parameters"`
	// 备注
	Remark string `json:"remark"`
}
