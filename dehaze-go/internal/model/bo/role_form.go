package bo

// RoleFormBO 角色表单对象
type RoleFormBO struct {
	// 角色ID
	ID *int64 `json:"id"`
	// 角色名称
	Name string `json:"name" binding:"required,min=2,max=64,no_xss"`
	// 角色编码
	Code string `json:"code" binding:"required,max=32,no_xss"`
	// 排序
	Sort int `json:"sort" binding:"min=0"`
	// 角色状态(1-正常；0-停用)
	Status int8 `json:"status" binding:"oneof=0 1"`
	// 数据权限（创建时必填，指针类型以区分"未选择"与合法值 0）
	DataScope *int8 `json:"dataScope" binding:"omitempty,min=0,max=3"`
}
