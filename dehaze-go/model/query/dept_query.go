package query

// DeptQuery 部门查询对象
type DeptQuery struct {
	// 关键字(部门名称)
	Keywords string `json:"keywords"`
	// 状态(1->正常；0->禁用)
	Status *int `json:"status"`
}