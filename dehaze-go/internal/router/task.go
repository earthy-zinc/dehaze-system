package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterTaskRoutes(rg *gin.RouterGroup, taskApi *api.SysTaskApi) {
	taskRouterGroup := rg.Group("/tasks")

	{
		// 读操作
		taskRouterGroup.GET("", taskApi.GetTaskPage)       // 任务分页列表
		taskRouterGroup.GET("/:id", taskApi.GetTaskById)    // 任务详情

		// 写操作 - 需要权限校验
		// 注意：任务创建/重试不加 AntiRepeat，因为用户可能对同一数据集重复导出，
		// 每次生成独立 taskId，PublishTask 内部已有分布式锁防重复发布
		taskRouterGroup.POST("", middleware.Permission("sys:task:add"), taskApi.CreateTask)    // 创建任务
		taskRouterGroup.POST("/:id/retry", middleware.Permission("sys:task:add"), taskApi.RetryTask) // 重试失败任务
		taskRouterGroup.DELETE("/:id", middleware.Permission("sys:task:delete"), taskApi.CancelTask) // 取消/删除任务
	}
}
