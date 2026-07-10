package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

func RegisterFileRoutes(rg *gin.RouterGroup, fileApi *api.SysFileApi) {
	fileRouterGroup := rg.Group("/files")

	{
		fileRouterGroup.POST("", fileApi.UploadFile)                       // 文件上传
		fileRouterGroup.DELETE("", fileApi.DeleteFile)                     // 文件删除
		fileRouterGroup.GET("/check", fileApi.CheckFile)                   // 文件校验
		fileRouterGroup.GET("/page", fileApi.GetFilePage)                  // 分页查询
		fileRouterGroup.GET("/download/*objectName", fileApi.DownloadFile) // 文件下载
		fileRouterGroup.GET("/:fileId", fileApi.GetFileDetail)             // 文件详情
	}
}
