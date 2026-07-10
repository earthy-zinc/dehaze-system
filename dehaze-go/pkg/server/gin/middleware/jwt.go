package middleware

import (
	"context"
	"errors"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/security"
)

func JWTAuth() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 我们这里jwt鉴权取头部信息 x-token 登录时回返回token信息 这里前端需要把token存储到cookie或者本地localStorage中 不过需要跟后端协商过期时间 可以约定刷新令牌或者重新登录
		token := security.GetToken(c)
		if token == "" {
			common.NoAuth("未登录或非法访问，请登录", c)
			c.Abort()
			return
		}

		if isBlacklist(token) {
			common.NoAuth("您的帐户异地登陆或令牌失效", c)
			security.ClearToken(c)
			c.Abort()
			return
		}

		j := security.NewJWT()
		// parseToken 解析token包含的信息
		claims, err := j.ParseToken(token)
		if err != nil {
			if errors.Is(err, security.ErrTokenExpired) {
				common.NoAuth("登录已过期，请重新登录", c)
				security.ClearToken(c)
				c.Abort()
				return
			}
			common.NoAuth(err.Error(), c)
			security.ClearToken(c)
			c.Abort()
			return
		}

		// 已登录用户被管理员禁用 需要使该用户的jwt失效 此处比较消耗性能 如果需要 请自行打开
		// 用户被删除的逻辑 需要优化 此处比较消耗性能 如果需要 请自行打开

		//if user, err := userService.FindUserByUuid(claims.UUID.String()); err != nil || user.Enable == 2 {
		//	_ = jwtService.JsonInBlacklist(system.JwtBlacklist{Jwt: token})
		//	common.FailWithDetailed(gin.H{"reload": true}, err.Error(), c)
		//	c.Abort()
		//}
		c.Set("claims", claims)
		c.Next()
	}
}

func isBlacklist(jwt string) bool {
	cacheClient := cache.GetCache()

	// 解析Token获取jti
	j := security.NewJWT()
	claims, err := j.ParseToken(jwt)
	if err != nil {
		// Token解析失败，视为无效Token
		return false
	}

	jti := claims.ID
	if jti == "" {
		return false
	}

	exists, err := cacheClient.Exists(context.Background(), common.BlacklistPrefix+jti)
	if err != nil {
		// 缓存服务异常时记录日志，但不阻止请求（保证服务可用性）
		// 注意：这里不返回true，避免因缓存故障导致所有请求被拒绝
		logger.Error("检查Token黑名单失败", zap.Error(err))
		return false
	}

	return exists
}
