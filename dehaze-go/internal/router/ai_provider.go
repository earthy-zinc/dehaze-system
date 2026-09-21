package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

// RegisterAiProviderRoutes AI 模型供应商与 API Key（/api/v1/ai/providers），
// 除 enabled 列表外均需 ai:model:manage。路径参数统一 :id。
func RegisterAiProviderRoutes(rg *gin.RouterGroup, providerApi *api.AiProviderApi) {
	group := rg.Group("/ai/providers")
	manage := middleware.Permission("ai:model:manage")
	{
		group.GET("", manage, providerApi.ListProviders)
		group.GET("/enabled", providerApi.ListEnabledProviders)
		group.POST("", manage, providerApi.CreateProvider)
		group.PUT("/:id", manage, providerApi.UpdateProvider)
		group.DELETE("/:id", manage, providerApi.DeleteProvider)
		group.GET("/:id/keys", manage, providerApi.ListProviderKeys)
		group.POST("/:id/keys", manage, providerApi.CreateProviderKey)
		group.PUT("/:id/keys/:id", manage, providerApi.UpdateProviderKey)
		group.DELETE("/:id/keys/:id", manage, providerApi.DeleteProviderKey)
		group.POST("/:id/circuit/close", manage, providerApi.CloseProviderCircuit)
	}
}
