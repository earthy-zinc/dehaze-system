package read

import "time"

// UserPage 用户分页读模型
type UserPage struct {
	ID          int64     `json:"id"`
	Username    string    `json:"username"`
	Nickname    string    `json:"nickname"`
	Mobile      string    `json:"mobile"`
	GenderLabel string    `json:"genderLabel"`
	Avatar      string    `json:"avatar"`
	Email       string    `json:"email"`
	Status      int8      `json:"status"`
	DeptName    string    `json:"deptName"`
	RoleNames   string    `json:"roleNames"`
	CreateTime  time.Time `json:"createTime"`
}
