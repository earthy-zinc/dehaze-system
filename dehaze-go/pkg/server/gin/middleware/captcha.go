package middleware

import (
	"strings"

	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// CaptchaConfig 验证码中间件配置
type CaptchaConfig struct {
	// EnabledPaths 需要提取验证码参数的路径列表（支持部分匹配）
	EnabledPaths []string
}

// Captcha 验证码参数提取中间件
// 仅负责从请求中提取验证码参数并存入Context，校验逻辑由Service层处理
func Captcha(config CaptchaConfig) gin.HandlerFunc {
	return func(c *gin.Context) {
		path := c.Request.URL.Path
		shouldExtract := false
		for _, enabledPath := range config.EnabledPaths {
			if strings.Contains(path, enabledPath) {
				shouldExtract = true
				break
			}
		}

		if !shouldExtract {
			c.Next()
			return
		}

		captchaCode := c.PostForm("captchaCode")
		captchaKey := c.PostForm("captchaKey")

		if captchaCode != "" && captchaKey != "" {
			c.Set("captchaCode", captchaCode)
			c.Set("captchaKey", captchaKey)
			logger.Debug("验证码参数已提取",
				zap.String("captchaKey", captchaKey),
				zap.String("ip", c.ClientIP()))
		}

		c.Next()
	}
}
