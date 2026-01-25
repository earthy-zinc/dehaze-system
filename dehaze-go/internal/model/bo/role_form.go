package bo

// RoleFormBO 角色表单对象
type RoleFormBO struct {
	// 角色ID
	ID *int64 `json:"id"`
	// 角色名称
	Name string `json:"name" binding:"required"`
	// 角色编码
	Code string `json:"code" binding:"required"`
	// 排序
	Sort int `json:"sort"`
	// 角色状态(1-正常；0-停用)
	Status int8 `json:"status"`
	// 数据权限
	DataScope int8 `json:"dataScope"`
}
