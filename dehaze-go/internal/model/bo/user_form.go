package bo

// UserFormBO 用户表单业务对象
type UserFormBO struct {
	// 用户ID
	ID int64 `json:"id"`
	// 用户名
	Username string `json:"username" binding:"required,min=2,max=50"`
	// 用户昵称
	Nickname string `json:"nickname" binding:"required,max=64,no_xss"`
	// 手机号
	Mobile string `json:"mobile" binding:"omitempty,len=11"`
	// 性别(1:男;2:女)
	Gender int8 `json:"gender" binding:"oneof=0 1 2"`
	// 用户头像
	Avatar string `json:"avatar" binding:"omitempty,max=255"`
	// 用户邮箱
	Email string `json:"email" binding:"omitempty,email,max=128"`
	// 状态(1:启用;0:禁用)
	Status int8 `json:"status" binding:"oneof=0 1"`
	// 部门ID
	DeptID int64 `json:"deptId" binding:"required,gt=0"`
	// 角色ID集合
	RoleIds []int64 `json:"roleIds" binding:"required,min=1"`
}
