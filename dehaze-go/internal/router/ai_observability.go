package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

// RegisterAiObservabilityRoutes 注册 AI 可观测性路由（/api/v1/ai/observability，8 端点）。
//
// literal 路径（/traces/export）注册在参数路径（/traces/:traceId）之前，
// 避免 export 被当作 traceId 解析。管理端审计权限（ai:conversation:audit）在 handler 内校验。
func RegisterAiObservabilityRoutes(rg *gin.RouterGroup, observabilityApi *api.AiObservabilityApi) {
	group := rg.Group("/ai/observability")
	{
		group.GET("/summary", observabilityApi.GetSummary)
		group.GET("/traces", observabilityApi.ListTraces)
		group.GET("/traces/export", observabilityApi.ExportTraces)
		group.GET("/traces/:traceId", observabilityApi.GetTrace)
		group.GET("/conversations/:id/timeline", observabilityApi.GetConversationTimeline)
		group.GET("/conversations/:id/timeline/export", observabilityApi.ExportConversationTimeline)
		group.GET("/costs", observabilityApi.GetCosts)
		group.GET("/trends", observabilityApi.GetTrends)
	}
}
