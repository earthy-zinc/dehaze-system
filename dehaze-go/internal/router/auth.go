package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

// 登录/注册限流安全默认值：10次/60秒，与 Java @RateLimit 对齐
const defaultLoginIPLimitCount = 10

func RegisterNoAuthRoutes(rg *gin.RouterGroup, authApi *api.AuthApi) gin.IRoutes {
	authRouter := rg.Group("auth")
	{
		// login/register 限流阈值可通过 system.login-ip-limit-count 配置，未配置时使用安全默认值 10 次/分钟
		loginLimit := config.GetConfig().System.LoginIPLimitCount
		if loginLimit <= 0 {
			loginLimit = defaultLoginIPLimitCount
		}
		authRouter.POST("login", middleware.CustomIPRateLimiter(60, loginLimit, "login"), authApi.Login)
		authRouter.POST("register", middleware.CustomIPRateLimiter(60, loginLimit, "register"), authApi.Register)
		authRouter.GET("captcha", authApi.Captcha)
	}
	return authRouter
}

func RegisterAuthRoutes(rg *gin.RouterGroup, authApi *api.AuthApi) gin.IRoutes {
	authRouter := rg.Group("auth")
	{
		authRouter.POST("logout", authApi.Logout)
		authRouter.GET("me", authApi.GetAuthInfo)
	}
	return authRouter
}
