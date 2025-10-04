package router

import "github.com/gin-gonic/gin"

type UserRouter struct{}

func (r *UserRouter) InitUserRouter(Router *gin.RouterGroup) (R gin.IRoutes) {
	userRouter := Router.Group("user")
	{
		userRouter.POST("login", authApi.Login)
		userRouter.POST("captcha", authApi.Captcha)
	}
	return userRouter
}
