package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

// RegisterAiUserPreferenceRoutes 用户偏好默认模型（/api/v1/ai/user-preference），登录即可读写，无权限码。
func RegisterAiUserPreferenceRoutes(rg *gin.RouterGroup, userPreferenceApi *api.AiUserPreferenceApi) {
	group := rg.Group("/ai/user-preference")
	{
		group.GET("", userPreferenceApi.Get)
		group.PUT("", userPreferenceApi.Update)
	}
}

// RegisterAiModelRoutes AI 模型注册表与用户售价（/api/v1/ai/models），
// 除 enabled 列表外均需 ai:model:manage。model_id 是含 `/` 的厂商业务键，
// 不能作路径段（会被当路径分隔符），故一律以 query 参数 modelId 传递；
// 仅价格版本的 price_id 用路径参数 :priceId（与 go-proxy 转发路由共节点时通配符名须一致）。
func RegisterAiModelRoutes(rg *gin.RouterGroup, modelApi *api.AiModelApi) {
	group := rg.Group("/ai/models")
	manage := middleware.Permission("ai:model:manage")
	{
		group.GET("", manage, modelApi.ListModels)
		group.GET("/enabled", modelApi.ListEnabledModels)
		group.POST("", manage, modelApi.CreateModel)
		group.PUT("", manage, modelApi.UpdateModel)
		group.DELETE("", manage, modelApi.DeleteModel)
		group.GET("/prices", manage, modelApi.ListModelPrices)
		group.POST("/prices", manage, modelApi.CreateModelPrice)
		group.PUT("/prices/:priceId", manage, modelApi.UpdateModelPrice)
		group.DELETE("/prices/:priceId", manage, modelApi.DeleteModelPrice)
	}
}
