package router

import "github.com/gin-gonic/gin"

type AuthRouter struct{}

func (r *AuthRouter) InitAuthRouter(Router *gin.RouterGroup) (R gin.IRoutes) {
	authRouter := Router.Group("auth")
	{
		authRouter.POST("login", authApi.Login)
		authRouter.POST("captcha", authApi.Captcha)
	}
	return authRouter

}
