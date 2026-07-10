package common

const (
	CaptchaCodePrefix = "captcha_code:"

	RolePermsPrefix = "role_perms:"

	// BlacklistPrefix Token黑名单前缀，存储jti（Token ID）而非完整Token
	// 格式：token:blacklist:{jti}
	BlacklistPrefix = "token:blacklist:"
)
