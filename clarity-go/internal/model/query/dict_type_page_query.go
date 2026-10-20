package query

// DictTypePageQuery 字典类型分页查询对象
type DictTypePageQuery struct {
	// 关键字(类型名称/类型编码)
	Keywords string `json:"keywords"`
	// 页码
	PageNum int `json:"pageNum"`
	// 每页条数
	PageSize int `json:"pageSize"`
}
