package vo

// UserImportVO 用户导入对象
type UserImportVO struct {
	// 用户名
	Username string `json:"username"`
	// 昵称
	Nickname string `json:"nickname"`
	// 性别
	Gender string `json:"gender"`
	// 部门名称
	DeptName string `json:"deptName"`
	// 手机号码
	Mobile string `json:"mobile"`
	// 邮箱
	Email string `json:"email"`
	// 状态
	Status string `json:"status"`
}
