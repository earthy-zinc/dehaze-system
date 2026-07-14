package common

const (
	CaptchaCodePrefix = "captcha_code:"

	// BlacklistPrefix Token黑名单前缀，存储jti（Token ID）而非完整Token
	// 格式：token:blacklist:{jti}
	BlacklistPrefix = "token:blacklist:"
)

const SystemUserID int64 = 0
