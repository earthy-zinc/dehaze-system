package router

import (
	"github.com/earthyzinc/clarity-go/internal/api"
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
		// 不挂防重：上传为 multipart（无法有效摘要），且"连续上传多个文件"是合法操作
		fileRouterGroup.POST("", fileApi.UploadFile)   // 文件上传
		fileRouterGroup.DELETE("", fileApi.DeleteFile) // 文件删除
	}
}
