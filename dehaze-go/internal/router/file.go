package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterFileRoutes(rg *gin.RouterGroup, fileApi *api.SysFileApi) {
	fileRouterGroup := rg.Group("/files")

	{
		// 读操作 - 无需额外权限（下载/详情/列表在 API 层做归属校验：普通用户仅限本人文件）
		fileRouterGroup.GET("/check", fileApi.CheckFile)       // 文件校验（秒传预检，无归属语义）
		fileRouterGroup.GET("/page", fileApi.GetFilePage)      // 分页查询
		fileRouterGroup.GET("/:fileId", fileApi.GetFileDetail) // 文件详情
		fileRouterGroup.GET("/download/*objectName", fileApi.DownloadFile)

		// 写操作 - 规范：文件管理模块不实现权限控制，仅需登录态（对齐 Java/Python）
		fileRouterGroup.POST("", middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 5, IncludeBody: true}), fileApi.UploadFile) // 文件上传
		fileRouterGroup.DELETE("", fileApi.DeleteFile)                                                                                 // 文件删除
	}
}
