package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterEvaluationRoutes(rg *gin.RouterGroup, evaluationApi *api.SysEvaluationApi) {
	evaluationGroup := rg.Group("/evaluation")

	evaluationGroup.POST("", middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 5}), evaluationApi.Evaluate) // 执行效果评估
	evaluationGroup.GET("/:id", evaluationApi.GetEvaluationLog)                                                     // 查询评估状态
	evaluationGroup.GET("/logs", evaluationApi.ListEvaluationLogs)                                                  // 评估日志列表
}
