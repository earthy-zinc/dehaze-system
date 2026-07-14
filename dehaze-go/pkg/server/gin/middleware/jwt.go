package middleware

import (
	"context"
	"net/http"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/security"
)

// JWTAuth JWT 认证中间件
func JWTAuth() gin.HandlerFunc {
	return func(c *gin.Context) {
		token := security.GetToken(c)
		if token == "" {
			unauthorized(c)
			return
		}

		if isBlacklist(c.Request.Context(), token) {
			security.ClearToken(c)
			unauthorized(c)
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
		Data: map[string]interface{}{},
		Msg:  common.TOKEN_INVALID.Msg,
	})
	c.Abort()
}

func isBlacklist(ctx context.Context, jwt string) bool {
	cacheClient := cache.GetCache()

	j := security.NewJWT()
	claims, err := j.ParseToken(jwt)
	if err != nil {
		return false
	}

	jti := claims.ID
	if jti == "" {
		return false
	}

	exists, err := cacheClient.Exists(ctx, common.BlacklistPrefix+jti)
	if err != nil {
		logger.Error("检查Token黑名单失败", zap.Error(err))
		return false
	}

	return exists
}
