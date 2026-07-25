package bo

type LoginRequest struct {
	Username    string `json:"username" binding:"required"`
	Password    string `json:"password" binding:"required"`
	CaptchaCode string `json:"captchaCode"`
	CaptchaKey  string `json:"captchaKey"`
	RememberMe  *bool  `json:"rememberMe"`
}

type LogoutRequest struct {
	Token string `json:"token"`
}
