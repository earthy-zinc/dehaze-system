package dto

// LoginResult 登录响应对象
type LoginResult struct {
	// 访问token
	AccessToken string `json:"accessToken"`
	// token 类型
	TokenType string `json:"tokenType"`
	// 刷新token
	RefreshToken string `json:"refreshToken"`
	// 过期时间(单位：毫秒)
	Expires int64 `json:"expires"`
	// 用户信息
	User *LoginUser `json:"user"`
}

// LoginUser 登录返回的用户信息
type LoginUser struct {
	ID       int64  `json:"id"`
	Username string `json:"username"`
	Nickname string `json:"nickname"`
}
