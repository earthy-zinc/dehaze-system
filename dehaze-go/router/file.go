package router

import (
	"github.com/earthyzinc/dehaze-go/api"
	"github.com/earthyzinc/dehaze-go/middleware"
	"github.com/gin-gonic/gin"
)

type FileRouter struct{}

func (fileRouter *FileRouter) InitFileRouter(routerGroup *gin.RouterGroup) {
	fileApi := api.ApiGroupApp.SysFileApi
	fileRouterGroup := routerGroup.Group("/files").
		Use(middleware.JWTAuth())

	{
		fileRouterGroup.POST("", fileApi.UploadFile)                       // 文件上传
		fileRouterGroup.DELETE("", fileApi.DeleteFile)                     // 文件删除
		fileRouterGroup.GET("/check", fileApi.CheckFile)                   // 文件校验
		fileRouterGroup.GET("/download/*objectName", fileApi.DownloadFile) // 文件下载
	}
}
