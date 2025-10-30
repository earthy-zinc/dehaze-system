package dto

// UserAuthInfo 用户认证信息
type UserAuthInfo struct {
	UserId    int64    `json:"userId"`
	Username  string   `json:"username"`
	Nickname  string   `json:"nickname"` // 补充缺失的Nickname字段
	DeptId    int64    `json:"deptId"`   // 修正类型为int64以匹配Java的Long类型
	Password  string   `json:"password"`
	Status    int8     `json:"status"` // 修正类型为int8以匹配Java的Integer类型（状态通常较小）
	Roles     []string `json:"roles"`
	Perms     []string `json:"perms"`
	DataScope int8     `json:"dataScope"` // 修正类型为int8
}
