package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

// RegisterAiKbRoutes AI 知识库 A 类端点（/api/v1/kb）。路径参数统一 :id。
// 创建/删除知识库、index-stats 依赖 ES 索引，文档写路径依赖向量入库，均由 go-proxy 转发。
func RegisterAiKbRoutes(rg *gin.RouterGroup, kbApi *api.AiKbApi) {
	group := rg.Group("/kb")
	manage := middleware.Permission("kb:manage")
	audit := middleware.Permission("kb:audit")
	{
		group.GET("", kbApi.ListKnowledgeBases)
		group.GET("/:id", kbApi.GetKnowledgeBase)
		group.PUT("/:id", manage, kbApi.UpdateKnowledgeBase)
		group.GET("/:id/documents", kbApi.ListDocuments)
		group.GET("/documents/:id", kbApi.GetDocument)
		group.GET("/documents/:id/chunks", kbApi.ListDocumentChunks)
		group.POST("/:id/retrieve/test-sets", audit, kbApi.CreateTestSet)
		group.GET("/:id/retrieve/test-sets", audit, kbApi.ListTestSets)
		group.GET("/:id/chunks/low-quality", audit, kbApi.ListLowQuality)
	}
}
