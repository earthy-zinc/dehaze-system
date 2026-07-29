package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterNoAuthRoutes(rg *gin.RouterGroup, authApi *api.AuthApi) gin.IRoutes {
	authRouter := rg.Group("auth")
	{
		// login/register 限流：10次/60秒，与 Java @RateLimit 对齐
		authRouter.POST("login", middleware.CustomIPRateLimiter(60, 10, "login"), authApi.Login)
		authRouter.POST("register", middleware.CustomIPRateLimiter(60, 10, "register"), authApi.Register)
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
