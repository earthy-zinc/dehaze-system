package vo

// UserInfoVO 用户登录视图对象
type UserInfoVO struct {
	// 用户ID
	UserId int64 `json:"userId"`
	// 用户名
	Username string `json:"username"`
	// 用户昵称
	Nickname string `json:"nickname"`
	// 头像地址
	Avatar string `json:"avatar"`
	// 用户角色编码集合
	Roles []string `json:"roles"`
	// 用户权限标识集合
	Perms []string `json:"perms"`
}
