package dto

// CaptchaResult 验证码响应对象
type CaptchaResult struct {
	// 验证码缓存key
	CaptchaKey string `json:"captchaKey"`
	// 验证码图片Base64字符串
	CaptchaBase64 string `json:"captchaBase64"`
}
