package read

import "time"

// UserPageWithRoles 用户分页读模型（含角色名称）
type UserPageWithRoles struct {
	ID         int64     `json:"id"`
	Username   string    `json:"username"`
	Nickname   string    `json:"nickname"`
	Mobile     string    `json:"mobile"`
	Gender     int8      `json:"gender"`
	Avatar     string    `json:"avatar"`
	Status     int8      `json:"status"`
	Email      string    `json:"email"`
	DeptName   string    `json:"deptName"`
	RoleNames  string    `json:"roleNames"`
	CreateTime time.Time `json:"createTime"`
}
