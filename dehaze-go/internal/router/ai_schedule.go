package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

// RegisterAiScheduleRoutes 注册 AI 定时任务（A 类）与运营统计。
//
// B 类端点（POST /ai/scheduled-tasks/:id/run 手动触发）由 go-proxy 的转发路由注册。
func RegisterAiScheduleRoutes(rg *gin.RouterGroup, scheduleApi *api.AiScheduleApi, usageApi *api.AiUsageApi) {
	schedules := rg.Group("/ai/scheduled-tasks")
	{
		schedules.GET("", scheduleApi.ListSchedules)
		schedules.POST("", scheduleApi.CreateSchedule)
		schedules.GET("/next-times", scheduleApi.PreviewNextTimes)
		schedules.GET("/:id", scheduleApi.GetSchedule)
		schedules.PUT("/:id", scheduleApi.UpdateSchedule)
		schedules.DELETE("/:id", scheduleApi.DeleteSchedule)
		schedules.PATCH("/:id/status", scheduleApi.SetScheduleStatus)
		schedules.GET("/:id/history", scheduleApi.ListRunHistory)
	}

	rg.GET("/ai/usage/stats", usageApi.GetUsageStats)
}
