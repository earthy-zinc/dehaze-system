package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

func RegisterFavoriteRoutes(rg *gin.RouterGroup, favoriteApi *api.FavoriteApi) {
	favRouter := rg.Group("/favorites")
	{
		// literal 路径优先注册
		favRouter.GET("/page", favoriteApi.GetPage)
		favRouter.GET("/count", favoriteApi.GetCount)
		// 带参数路径
		favRouter.GET("/:targetId/status", favoriteApi.GetStatus)
		favRouter.DELETE("/:ids", favoriteApi.DeleteByIDs)
		favRouter.POST("", favoriteApi.Add)
	}
}
