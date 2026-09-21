package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

// RegisterAiEvalRoutes 注册智能体评测集/样本与评测中心（A 类）。
//
// B 类端点（POST /ai/agents/:id/eval/runs、GET .../eval/runs、GET .../eval/tasks/:task_id）
// 由 go-proxy 的转发路由注册，此处不重复注册。
func RegisterAiEvalRoutes(rg *gin.RouterGroup, a *api.AiEvalApi) {
	agents := rg.Group("/ai/agents")
	{
		agents.POST("/:id/eval/datasets", a.CreateEvalDataset)
		agents.GET("/:id/eval/datasets", a.ListEvalDatasets)
		agents.PATCH("/:id/eval/datasets/:datasetId", a.UpdateEvalDataset)
		agents.DELETE("/:id/eval/datasets/:datasetId", a.DeleteEvalDataset)
		agents.POST("/:id/eval/datasets/:datasetId/samples", a.CreateEvalSample)
		agents.GET("/:id/eval/datasets/:datasetId/samples", a.ListEvalSamples)
		agents.PATCH("/:id/eval/samples/:sampleId", a.UpdateEvalSample)
		agents.DELETE("/:id/eval/samples/:sampleId", a.DeleteEvalSample)
	}

	center := rg.Group("/ai/eval-center")
	{
		center.GET("/overview", a.EvalOverview)
		center.GET("/trends", a.EvalTrends)
		center.GET("/judge-status", a.EvalJudgeStatus)
		center.GET("/reviews", a.EvalReviews)
		center.POST("/reviews/:id", a.SubmitEvalReview)
		center.GET("/runs/:id/compare", a.EvalRunCompare)
		center.GET("/runs/:id/samples/:sampleId", a.EvalReviewDetail)
	}
}
