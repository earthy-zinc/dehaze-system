package query

// AlgorithmQuery 算法列表查询对象。
// python `GET /algorithms` 只声明 keywords（无分页参数，未知参数静默忽略），
// 故分页字段不参与 gin 绑定——否则 `pageNum=abc` 会在 go 报 A0400、在 python 正常返回。
type AlgorithmQuery struct {
	Keywords string `json:"keywords" form:"keywords"` // 关键字（名称/类型模糊搜索）
	Type     string `json:"type" form:"type"`         // 模型类型筛选
	Status   *int8  `json:"status" form:"status"`     // 状态筛选（0-5六态）
	PageNum  int    `json:"pageNum"`
	PageSize int    `json:"pageSize"`
}

// AlgorithmCompareQuery 算法对比查询
type AlgorithmCompareQuery struct {
	IDs []int64 `json:"ids" form:"ids"` // 算法ID列表
}
