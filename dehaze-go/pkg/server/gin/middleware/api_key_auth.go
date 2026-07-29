package middleware

import (
	"context"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/earthyzinc/dehaze-go/pkg/security"
)

// ApiKeyAuthenticator 是 API Key 认证的校验函数签名，由 app 层注入具体实现。
type ApiKeyAuthenticator func(ctx context.Context, rawKey string) (*security.CustomClaims, error)

// ApiKeyAuth 是外部注入的 API Key 校验实现（默认 nil）。
var ApiKeyAuth ApiKeyAuthenticator

// ApiKeyAuthMiddleware 独立的 API Key 认证中间件，负责处理 Bearer dhak_* 格式的凭证。
// 应在 SessionAuth 之前注册。
func ApiKeyAuthMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		token := c.Request.Header.Get("Authorization")
		if token == "" {
			c.Next()
			return
		}
		if strings.HasPrefix(token, "Bearer ") {
			token = token[7:]
		}
		if !strings.HasPrefix(token, "dhak_") {
			c.Next()
			return
		}

		if ApiKeyAuth == nil {
			unauthorized(c)
			return
		}

		claims, err := ApiKeyAuth(c.Request.Context(), token)
		if err != nil {
			unauthorized(c)
			return
		}

		c.Set("claims", claims)
		c.Next()
	}
}
