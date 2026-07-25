package bo

// LoginRequest 登录请求参数
type LoginRequest struct {
	Username    string `json:"username" binding:"required"`
	Password    string `json:"password" binding:"required"`
	CaptchaCode string `json:"captchaCode"`
	CaptchaKey  string `json:"captchaKey"`
	RememberMe  *bool  `json:"rememberMe"`
}

// LogoutRequest 注销请求参数
type LogoutRequest struct {
	Token string `json:"token"`
}

// RefreshTokenRequest 刷新令牌请求参数
type RefreshTokenRequest struct {
	RefreshToken string `json:"refreshToken"`
}
