package vo

import "time"

// UserExportVO 用户导出视图对象
type UserExportVO struct {
	// 用户名
	Username string `json:"username"`
	// 用户昵称
	Nickname string `json:"nickname"`
	// 部门
	DeptName string `json:"deptName"`
	// 性别
	Gender string `json:"gender"`
	// 手机号码
	Mobile string `json:"mobile"`
	// 邮箱
	Email string `json:"email"`
	// 创建时间
	CreateTime time.Time `json:"createTime"`
}
