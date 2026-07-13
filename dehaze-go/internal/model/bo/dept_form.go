package bo

// DeptFormBO 部门表单对象
type DeptFormBO struct {
	// 部门ID
	ID *int64 `json:"id"`
	// 部门名称
	Name string `json:"name" binding:"required,min=1,max=64"`
	// 父部门ID
	ParentID int64 `json:"parentId"`
	// 状态(1:启用;0:禁用)
	Status int8 `json:"status" binding:"oneof=0 1"`
	// 排序(数字越小排名越靠前)
	Sort int `json:"sort" binding:"min=0"`
}
