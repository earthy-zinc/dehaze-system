package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

// RegisterAiConversationRoutes 注册 AI 会话/消息/反馈/产物（A 类非推理端点）。
//
// 路径参数统一用 :id，与 B 类转发路由（go-proxy 注册的 /ai/conversations/:id/messages 等）
// 在同一节点位置保持一致，否则 gin 启动 panic（wildcard segment conflicts）。
// literal 路径一律注册在 /:id 之前。
func RegisterAiConversationRoutes(rg *gin.RouterGroup, a *api.AiConversationApi) {
	conversations := rg.Group("/ai/conversations")
	{
		conversations.GET("", a.ListConversations)
		conversations.POST("", a.CreateConversation)
		conversations.GET("/trash", a.ListTrashConversations)
		conversations.POST("/batch", a.BatchConversations)
		conversations.GET("/:id", a.GetConversation)
		conversations.PATCH("/:id", a.UpdateConversation)
		conversations.DELETE("/:id", a.DeleteConversation)
		conversations.POST("/:id/restore", a.RestoreConversation)
		conversations.PUT("/:id/pin", a.PinConversation)
		conversations.PUT("/:id/unpin", a.UnpinConversation)
		conversations.PUT("/:id/read", a.ReadConversation)
		conversations.GET("/:id/export", a.ExportConversation)
		conversations.GET("/:id/messages", a.ListMessages)
		conversations.GET("/:id/messages/:messageId/branches", a.GetBranches)
		conversations.PUT("/:id/branches/:messageId", a.SwitchBranch)
		conversations.GET("/:id/artifacts", a.ListConversationArtifacts)
	}

	messages := rg.Group("/ai/messages")
	{
		messages.GET("/:id", a.GetMessage)
		messages.DELETE("/:id", a.DeleteMessage)
		messages.GET("/:id/feedback", a.GetFeedback)
		messages.POST("/:id/feedback", a.SubmitFeedback)
		messages.DELETE("/:id/feedback", a.RevokeFeedback)
		messages.GET("/:id/artifacts", a.ListMessageArtifacts)
	}

	artifacts := rg.Group("/ai/artifacts")
	{
		artifacts.GET("/by-ref", a.ListArtifactsByRef)
		artifacts.GET("/:id/detail", a.GetArtifactDetail)
	}
}

// RegisterAiMemoryRoutes 注册 AI 长期记忆（A 类）。
func RegisterAiMemoryRoutes(rg *gin.RouterGroup, a *api.AiMemoryApi) {
	memories := rg.Group("/ai/memories")
	{
		memories.GET("", a.ListMemories)
		memories.POST("", a.CreateMemory)
		memories.GET("/archived", a.ListArchived)
		memories.GET("/search", a.SearchMemories)
		memories.GET("/export", a.ExportMemories)
		memories.POST("/clear", a.ClearMemories)
		memories.POST("/restore", a.RestoreMemories)
		memories.PUT("/:id", a.UpdateMemory)
		memories.DELETE("/:id", a.DeleteMemory)
		memories.POST("/:id/unarchive", a.UnarchiveMemory)
	}
}
