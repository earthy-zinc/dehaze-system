package query

// RolePageQuery 角色分页查询对象
type RolePageQuery struct {
	// 关键字(角色名称/角色编码)
	Keywords string `json:"keywords"`
	// 页码
	PageNum int `json:"pageNum"`
	// 每页条数
	PageSize int `json:"pageSize"`
}
