package router

import (
	"github.com/earthyzinc/clarity-go/internal/api"
	"github.com/gin-gonic/gin"
)

func RegisterApiKeyRoutes(rg *gin.RouterGroup, apiKeyApi *api.ApiKeyApi) gin.IRoutes {
	authRouter := rg.Group("auth")
	{
		authRouter.POST("api-keys", apiKeyApi.CreateApiKey)
		authRouter.GET("api-keys", apiKeyApi.ListApiKeys)
		authRouter.DELETE("api-keys/:id", apiKeyApi.DeleteApiKey)
	}
	return authRouter
}
