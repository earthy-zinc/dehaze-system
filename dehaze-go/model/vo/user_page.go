package vo

import "time"

// UserPageVO 用户分页视图对象
type UserPageVO struct {
	// 用户ID
	ID int64 `json:"id"`
	// 用户名
	Username string `json:"username"`
	// 用户昵称
	Nickname string `json:"nickname"`
	// 手机号
	Mobile string `json:"mobile"`
	// 性别
	GenderLabel string `json:"genderLabel"`
	// 用户头像地址
	Avatar string `json:"avatar"`
	// 用户邮箱
	Email string `json:"email"`
	// 用户状态(1:启用;0:禁用)
	Status int8 `json:"status"`  // 修正类型为int8以匹配Java的Integer类型
	// 部门名称
	DeptName string `json:"deptName"`
	// 角色名称，多个使用英文逗号(,)分割
	RoleNames string `json:"roleNames"`
	// 创建时间
	CreateTime time.Time `json:"createTime"`
}