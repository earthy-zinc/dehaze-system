package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

// RegisterAiA2ARoutes A2A Agent Card 发现（A 类，登录态）。
// JSON-RPC 入口（POST /ai/agents/:id/a2a、POST /a2a、GET /.well-known/agent.json）
// 与兼容 API 由 go-proxy 转发，此处不注册。
func RegisterAiA2ARoutes(rg *gin.RouterGroup, a2aApi *api.AiA2AApi) {
	group := rg.Group("/ai/agents")
	{
		group.GET("/:id/a2a/.well-known/agent.json", a2aApi.AgentCard)
	}
}
