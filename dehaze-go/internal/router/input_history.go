package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

func RegisterImageInputRoutes(rg *gin.RouterGroup, api *api.SysInputHistoryApi) {
	historyGroup := rg.Group("/image-input/history")

	{
		// 创建历史记录不挂防重：输入建议类高频、幂等操作，5s 窗口内连续创建是合法行为
		historyGroup.GET("", api.ListHistory)                 // 分页查询历史记录
		historyGroup.GET("/:id", api.GetHistory)              // 历史记录详情
		historyGroup.POST("", api.CreateHistory)              // 创建历史记录
		historyGroup.DELETE("/:id", api.DeleteHistory)        // 删除单条
		historyGroup.DELETE("/batch", api.BatchDeleteHistory) // 批量删除
		historyGroup.DELETE("/clear", api.ClearHistory)       // 清空历史
	}
}
