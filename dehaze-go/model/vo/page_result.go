package vo

// PageResult 分页结果对象
type PageResult[T any] struct {
	// 数据列表
	List []T `json:"list"`
	// 总记录数
	Total int64 `json:"total"`
	// 页码
	PageNum int `json:"pageNum"`
	// 每页条数
	PageSize int `json:"pageSize"`
}
