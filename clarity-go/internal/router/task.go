package router

import (
	"github.com/earthyzinc/clarity-go/internal/api"
	"github.com/gin-gonic/gin"
)

func RegisterTaskRoutes(rg *gin.RouterGroup, taskApi *api.SysTaskApi) {
	taskRouterGroup := rg.Group("/tasks")

	{
		// 读操作
		taskRouterGroup.GET("", taskApi.GetTaskPage)                     // 任务分页列表
		taskRouterGroup.GET("/:id", taskApi.GetTaskById)                 // 任务详情
		taskRouterGroup.GET("/:id/download", taskApi.DownloadExportFile) // 下载任务结果

		// 写操作 - 无接口级权限标识（对齐 python/java 事实源：任务接口仅需登录态，
		// 操作权限由 service 层 create_by 归属校验保证）。
		// 历史上曾配置 sys:task:add / sys:task:delete，但这两个权限码未在 sys_menu /
		// sys_role_menu 种子中定义（任务菜单 perm=NULL），导致非 ROOT 管理员一律被误判
		// A0301（HTTP 403）而无法创建/重试/取消任务，已移除。
		// 注意：任务创建/重试不加 AntiRepeat，因为用户可能对同一数据集重复导出，
		// 每次生成独立 taskId，PublishTask 内部已有分布式锁防重复发布
		taskRouterGroup.POST("", taskApi.CreateTask)            // 创建任务
		taskRouterGroup.POST("/:id/retry", taskApi.RetryTask)   // 重试失败任务
		taskRouterGroup.POST("/:id/cancel", taskApi.CancelTask) // 取消任务
	}
}
