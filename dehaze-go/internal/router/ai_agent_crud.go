package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

// RegisterAiAgentRoutes 注册智能体管理（A 类：CRUD/启停/复制/关联/发布/版本/回滚）。
//
// B 类端点（POST /ai/agents/:id/test）由 go-proxy 的转发路由注册，此处不重复注册。
func RegisterAiAgentRoutes(rg *gin.RouterGroup, a *api.AiAgentApi) {
	agents := rg.Group("/ai/agents")
	{
		agents.GET("", a.ListAgents)
		agents.POST("", a.CreateAgent)
		agents.GET("/enabled", a.ListEnabledAgents)
		agents.GET("/config-defaults", a.GetAgentConfigDefaults)
		agents.GET("/:id", a.GetAgent)
		agents.PUT("/:id", a.UpdateAgent)
		agents.DELETE("/:id", a.DeleteAgent)
		agents.PATCH("/:id/status", a.SetAgentStatus)
		agents.POST("/:id/copy", a.CopyAgent)
		agents.PUT("/:id/skills", a.SetAgentSkills)
		agents.PUT("/:id/mcps", a.SetAgentMcps)
		agents.PUT("/:id/subagents", a.SetAgentSubagents)
		agents.GET("/:id/versions", a.ListAgentVersions)
		agents.GET("/:id/versions/diff", a.DiffAgentVersions)
		agents.GET("/:id/versions/:versionNo", a.GetAgentVersionDetail)
		agents.POST("/:id/versions/:versionNo/rollback", a.RollbackAgent)
	}

	endpoints := rg.Group("/ai/a2a/endpoints")
	{
		endpoints.GET("", a.ListEndpoints)
		endpoints.POST("", a.CreateEndpoint)
		endpoints.PATCH("/:id", a.UpdateEndpoint)
		endpoints.DELETE("/:id", a.DeleteEndpoint)
	}
}
