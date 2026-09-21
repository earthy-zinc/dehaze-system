package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

// RegisterAiModelRoutes AI 模型注册表与用户售价（/api/v1/ai/models），
// 除 enabled 列表外均需 ai:model:manage。路径参数统一 :id（仓库惯例，
// 与 go-proxy 的转发路由同节点相遇时通配符名必须一致）。
func RegisterAiModelRoutes(rg *gin.RouterGroup, modelApi *api.AiModelApi) {
	group := rg.Group("/ai/models")
	manage := middleware.Permission("ai:model:manage")
	{
		group.GET("", manage, modelApi.ListModels)
		group.GET("/enabled", modelApi.ListEnabledModels)
		group.POST("", manage, modelApi.CreateModel)
		group.PUT("/:id", manage, modelApi.UpdateModel)
		group.DELETE("/:id", manage, modelApi.DeleteModel)
		group.GET("/:id/prices", manage, modelApi.ListModelPrices)
		group.POST("/:id/prices", manage, modelApi.CreateModelPrice)
		group.PUT("/:id/prices/:id", manage, modelApi.UpdateModelPrice)
		group.DELETE("/:id/prices/:id", manage, modelApi.DeleteModelPrice)
	}
}
