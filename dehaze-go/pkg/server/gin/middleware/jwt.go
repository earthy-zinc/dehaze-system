package middleware

import (
	"context"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
)

var ApiKeyAuth func(ctx context.Context, rawKey string) (*security.CustomClaims, error)

// JWTAuth JWT 认证中间件
func JWTAuth() gin.HandlerFunc {
	return func(c *gin.Context) {
		token := security.GetToken(c)
		if token == "" {
			unauthorized(c)
			return
		}

		if strings.HasPrefix(token, "dhak_") {
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
			return
		}

		j := security.NewJWT()
		claims, err := j.ParseToken(token)
		if err != nil {
			security.ClearToken(c)
			unauthorized(c)
			return
		}

		c.Set("claims", claims)
		c.Next()
	}
}

// unauthorized 返回标准的 401 未授权响应
func unauthorized(c *gin.Context) {
	c.JSON(http.StatusUnauthorized, common.Response{
		Code: common.TOKEN_INVALID.Code,
		Data: map[string]any{},
		Msg:  common.TOKEN_INVALID.Msg,
	})
	c.Abort()
}

