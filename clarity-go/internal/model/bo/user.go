package bo

import "time"

// UserBO 用户业务对象
type UserBO struct {
	// 用户ID
	ID int64 `json:"id"`
	// 账户名
	Username string `json:"username"`
	// 昵称
	Nickname string `json:"nickname"`
	// 手机号
	Mobile string `json:"mobile"`
	// 性别(1->男；2->女)
	Gender int8 `json:"gender"` // 修正类型为int8以匹配Java的Integer类型
	// 头像URL
	Avatar string `json:"avatar"`
	// 邮箱
	Email string `json:"email"`
	// 状态: 1->启用;0->禁用
	Status int8 `json:"status"` // 修正类型为int8以匹配Java的Integer类型
	// 部门名称
	DeptName string `json:"deptName"`
	// 角色名称，多个使用英文逗号(,)分割
	RoleNames string `json:"roleNames"`
	// 创建时间
	CreateTime time.Time `json:"createTime"`
}
