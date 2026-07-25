package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

func RegisterNoAuthRoutes(rg *gin.RouterGroup, authApi *api.AuthApi) gin.IRoutes {
	authRouter := rg.Group("auth")
	{
		authRouter.POST("login", authApi.Login)
		authRouter.POST("register", authApi.Register)
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
