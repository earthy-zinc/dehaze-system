package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterImageInputRoutes(rg *gin.RouterGroup, api *api.SysInputHistoryApi) {
	historyGroup := rg.Group("/image-input/history")

	{
		historyGroup.GET("", api.ListHistory)                                                                   // 分页查询历史记录
		historyGroup.GET("/:id", api.GetHistory)                                                                // 历史记录详情
		historyGroup.POST("", middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 3}), api.CreateHistory) // 创建历史记录
		historyGroup.POST("/sync", api.SyncHistory)                                                             // 同步历史记录
		historyGroup.PUT("/:id", api.UpdateHistory)                                                             // 更新历史记录
		historyGroup.DELETE("/:id", api.DeleteHistory)                                                          // 删除单条
		historyGroup.DELETE("/batch", api.BatchDeleteHistory)                                                   // 批量删除
		historyGroup.DELETE("/clear", api.ClearHistory)                                                         // 清空历史
	}
}
