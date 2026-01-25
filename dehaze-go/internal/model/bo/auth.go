package bo

// LoginRequest 登录请求参数
type LoginRequest struct {
	// 用户名
	Username string `json:"username" binding:"required"`
	// 密码
	Password string `json:"password" binding:"required"`
	// 验证码
	CaptchaCode string `json:"captchaCode"`
	// 验证码Key
	CaptchaKey string `json:"captchaKey"`
}

// LogoutRequest 注销请求参数
type LogoutRequest struct {
	// Token
	Token string `json:"token"`
}

// RefreshTokenRequest 刷新令牌请求参数
type RefreshTokenRequest struct {
	// 刷新Token
	RefreshToken string `json:"refreshToken" binding:"required"`
}
