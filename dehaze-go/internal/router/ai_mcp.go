package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

// RegisterAiMcpRoutes 外部 MCP Server 管理（/api/v1/ai/mcp），管理操作统一 ai:mcp:manage。
// 路径参数统一 :id；health/tools/tools-test 为 B 类（go-proxy 转发），此处不注册。
func RegisterAiMcpRoutes(rg *gin.RouterGroup, mcpApi *api.AiMcpApi) {
	group := rg.Group("/ai/mcp")
	manage := middleware.Permission("ai:mcp:manage")
	{
		group.GET("/servers", manage, mcpApi.ListServers)
		group.POST("/servers", manage, mcpApi.CreateServer)
		group.GET("/servers/:id", manage, mcpApi.GetServer)
		group.PUT("/servers/:id", manage, mcpApi.UpdateServer)
		group.DELETE("/servers/:id", manage, mcpApi.DeleteServer)
		group.PATCH("/servers/:id/status", manage, mcpApi.SwitchServerStatus)
		group.GET("/servers/:id/namespaces", manage, mcpApi.ListNamespaces)
		group.PUT("/servers/:id/namespaces", manage, mcpApi.UpdateNamespaces)
		group.PUT("/servers/:id/credentials", manage, mcpApi.UpdateCredentials)
		group.GET("/market", manage, mcpApi.GetMarket)
		group.GET("/calls", manage, mcpApi.ListCalls)
	}
}
