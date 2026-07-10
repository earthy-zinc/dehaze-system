package read

import "time"

// UserExport 用户导出读模型
type UserExport struct {
	Username    string    `json:"username"`
	Nickname    string    `json:"nickname"`
	DeptName    string    `json:"deptName"`
	Gender      string    `json:"gender"`
	Mobile      string    `json:"mobile"`
	Email       string    `json:"email"`
	StatusLabel string    `json:"statusLabel"`
	CreateTime  time.Time `json:"createTime"`
}
