package query

// AlgorithmQuery 算法分页查询对象
type AlgorithmQuery struct {
	// 关键字
	Keywords string `json:"keywords" form:"keywords"`
	// 页码
	PageNum int `json:"pageNum" form:"pageNum"`
	// 每页条数
	PageSize int `json:"pageSize" form:"pageSize"`
}
