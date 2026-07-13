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
		taskRouterGroup.POST("", middleware.Permission("sys:task:add"), taskApi.CreateTask)    // 创建任务
		taskRouterGroup.DELETE("/:id", middleware.Permission("sys:task:delete"), taskApi.CancelTask) // 取消/删除任务
	}
}
