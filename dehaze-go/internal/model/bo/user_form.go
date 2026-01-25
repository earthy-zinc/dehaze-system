package bo

// UserFormBO 用户表单业务对象
type UserFormBO struct {
	// 用户ID
	ID int64 `json:"id"`
	// 用户名
	Username string `json:"username"`
	// 用户昵称
	Nickname string `json:"nickname"`
	// 手机号
	Mobile string `json:"mobile"`
	// 性别(1:男;2:女)
	Gender int8 `json:"gender"` // 修正类型为int8以匹配Java的Integer类型
	// 用户头像
	Avatar string `json:"avatar"`
	// 用户邮箱
	Email string `json:"email"`
	// 状态(1:启用;0:禁用)
	Status int8 `json:"status"` // 修正类型为int8以匹配Java的Integer类型
	// 部门ID
	DeptID int64 `json:"deptId"` // 修正类型为int64以匹配Java的Long类型
	// 角色ID集合
	RoleIds []int64 `json:"roleIds"` // 修正类型为int64以匹配Java的Long类型
}
