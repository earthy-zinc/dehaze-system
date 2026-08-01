package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

func RegisterCompareRoutes(rg *gin.RouterGroup, compareApi *api.CompareApi) {
	compareGroup := rg.Group("/compare")

	compareGroup.POST("/report", compareApi.GenerateReport)       // 生成对比报告（异步）
	compareGroup.GET("/report/:taskId", compareApi.GetOrDownloadReport) // 查询状态/下载报告
}
