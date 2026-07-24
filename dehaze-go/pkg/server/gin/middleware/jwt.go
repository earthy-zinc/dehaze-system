package middleware

import (
	"context"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
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

		// 检查黑名单：复用已解析的 claims，避免重复解析 token
		if isBlacklistedByClaims(c.Request.Context(), claims.ID) {
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

// isBlacklistedByClaims 根据 jti 检查 token 是否已加入黑名单
// 调用方需先解析出 claims 再传入，避免重复 ParseToken
// 安全策略：fail-closed。当缓存不可用或检查失败时（如 Redis 宕机），
// 视为 token 已失效（返回 true），拒绝请求，避免放行已注销的 token
func isBlacklistedByClaims(ctx context.Context, jti string) bool {
	if jti == "" {
		return false
	}
	cacheClient := cache.GetCache()
	if cacheClient == nil {
		logger.Error("检查Token黑名单失败：缓存不可用，按 fail-closed 拒绝")
		return true
	}
	exists, err := cacheClient.Exists(ctx, common.BlacklistPrefix+jti)
	if err != nil {
		logger.Error("检查Token黑名单失败，按 fail-closed 拒绝", zap.Error(err))
		return true
	}
	return exists
}
