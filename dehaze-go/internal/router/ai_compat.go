package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

// RegisterAiCompatAuditRoutes AI 兼容调用审计查询（/api/v1/ai/compat/calls，当前用户）
func RegisterAiCompatAuditRoutes(rg *gin.RouterGroup, compatApi *api.AiCompatAuditApi) {
	group := rg.Group("/ai/compat")
	{
		group.GET("/calls", compatApi.ListCalls)
	}
}
