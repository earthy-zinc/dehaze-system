package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

func RegisterPredictionRoutes(rg *gin.RouterGroup, predictionApi *api.SysPredictionApi) {
	predictionGroup := rg.Group("/prediction")

	predictionGroup.POST("", predictionApi.Predict)                 // 执行去雾预测
	predictionGroup.GET("/quota", predictionApi.GetQuota)           // 查询剩余次数（必须在 /:id 前）
	predictionGroup.POST("/batch", predictionApi.BatchPredict)      // 批量去雾预测（必须在 /:id 前）
	predictionGroup.GET("/logs", predictionApi.ListPredictionLogs)  // 预测日志列表
	predictionGroup.GET("/:id", predictionApi.GetPredictionLog)     // 查询预测状态
}
