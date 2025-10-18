package bo

// DatasetFormBO 数据集表单业务对象
type DatasetFormBO struct {
	// 数据集ID
	ID *int64 `json:"id"`
	// 父数据集ID
	ParentID int64 `json:"parentId" validate:"required"`
	// 数据集类型
	Type string `json:"type"`
	// 数据集名称
	Name string `json:"name"`
	// 数据集描述
	Description string `json:"description"`
	// 数据集存储路径
	Path string `json:"path"`
	// 状态(1:正常;0:禁用)
	Status int8 `json:"status"`
}