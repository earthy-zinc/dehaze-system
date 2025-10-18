package query

// DictPageQuery 字典数据项分页查询对象
type DictPageQuery struct {
	// 关键字(字典项名称)
	Keywords string `json:"keywords"`
	// 字典类型编码
	TypeCode string `json:"typeCode"`
	// 页码
	PageNum int `json:"pageNum"`
	// 每页条数
	PageSize int `json:"pageSize"`
}