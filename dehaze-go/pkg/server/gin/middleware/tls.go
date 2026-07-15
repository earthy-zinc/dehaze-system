package middleware

import (
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/gin-gonic/gin"
	"github.com/unrolled/secure"
	"go.uber.org/zap"
)

// LoadTls HTTPS 重定向中间件
// SSLHost 从配置读取（System.TlsHost），未配置时禁用 SSLHost 限制（保留 SSL 重定向）
func LoadTls() gin.HandlerFunc {
	return func(c *gin.Context) {
		opts := secure.Options{SSLRedirect: true}
		if host := config.GetConfig().System.TlsHost; host != "" {
			opts.SSLHost = host
		}
		middleware := secure.New(opts)
		if err := middleware.Process(c.Writer, c.Request); err != nil {
			logger.Error("TLS 中间件处理失败", zap.Error(err))
			c.Abort()
			return
		}
		c.Next()
	}
}
