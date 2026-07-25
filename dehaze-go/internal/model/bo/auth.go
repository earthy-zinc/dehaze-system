package bo

type LoginRequest struct {
	Username    string `json:"username" binding:"required"`
	Password    string `json:"password" binding:"required"`
	CaptchaCode string `json:"captchaCode"`
	CaptchaKey  string `json:"captchaKey"`
	RememberMe  *bool  `json:"rememberMe"`
}

type RegisterRequest struct {
	Username    string `json:"username" binding:"required,min=3,max=32"`
	Password    string `json:"password" binding:"required,min=6,max=20"`
	Nickname    string `json:"nickname" binding:"required,max=64"`
	CaptchaCode string `json:"captchaCode" binding:"required"`
	CaptchaKey  string `json:"captchaKey" binding:"required"`
}

type LogoutRequest struct {
	Token string `json:"token"`
}
