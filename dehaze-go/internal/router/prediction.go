package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterPredictionRoutes(rg *gin.RouterGroup, predictionApi *api.SysPredictionApi) {
	predictionGroup := rg.Group("/prediction")

	predictionGroup.POST("", middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 5}), predictionApi.Predict) // 执行去雾预测
	predictionGroup.GET("/:id", predictionApi.GetPredictionLog)                                                    // 查询预测状态
	predictionGroup.GET("/logs", predictionApi.ListPredictionLogs)                                                 // 预测日志列表
}
