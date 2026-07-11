package query

// AlgorithmQuery 算法分页查询对象
type AlgorithmQuery struct {
	Keywords string `json:"keywords" form:"keywords"` // 关键字（名称/类型模糊搜索）
	Type     string `json:"type" form:"type"`         // 模型类型筛选
	Status   *int8  `json:"status" form:"status"`     // 状态筛选（0-5六态）
	PageNum  int    `json:"pageNum" form:"pageNum"`   // 页码
	PageSize int    `json:"pageSize" form:"pageSize"` // 每页条数
}

// AlgorithmCompareQuery 算法对比查询
type AlgorithmCompareQuery struct {
	IDs []int64 `json:"ids" form:"ids"` // 算法ID列表
}
