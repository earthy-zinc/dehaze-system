package router

import "github.com/gin-gonic/gin"

type FileRouter struct{}

func (r *FileRouter) InitFileRouter(Router *gin.RouterGroup) (R gin.IRoutes) {
	fileRouter := Router.Group("file")
	{
		fileRouter.POST("login", authApi.Login)
		fileRouter.POST("captcha", authApi.Captcha)
	}
	return fileRouter
}
