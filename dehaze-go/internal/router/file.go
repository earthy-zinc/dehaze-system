package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterFileRoutes(rg *gin.RouterGroup, fileApi *api.SysFileApi) {
	fileRouterGroup := rg.Group("/files")

	{
		// 读操作 - 无需额外权限
		fileRouterGroup.GET("/check", fileApi.CheckFile)                   // 文件校验
		fileRouterGroup.GET("/page", fileApi.GetFilePage)                  // 分页查询
		fileRouterGroup.GET("/download/*objectName", fileApi.DownloadFile) // 文件下载
		fileRouterGroup.GET("/:fileId", fileApi.GetFileDetail)             // 文件详情

		// 写操作 - 需要权限校验 + 防重复提交
		fileRouterGroup.POST("", middleware.Permission("sys:file:add"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 5}), fileApi.UploadFile)         // 文件上传
		fileRouterGroup.DELETE("", middleware.Permission("sys:file:delete"), fileApi.DeleteFile)     // 文件删除
	}
}
