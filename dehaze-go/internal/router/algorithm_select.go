package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

// RegisterAlgorithmSelectRoutes 注册算法选择模块路由
// 基础路径: /api/v1/algorithms/select
// 注意：literal 路径必须注册在 /:id 之前以避免路由冲突
func RegisterAlgorithmSelectRoutes(rg *gin.RouterGroup, selectApi *api.AlgorithmSelectApi) {
	selectRouter := rg.Group("/algorithms/select")
	{
		// 静态路径优先注册
		selectRouter.GET("/tree", selectApi.GetTree)
		selectRouter.GET("/search", selectApi.Search)
		selectRouter.POST("/compare", selectApi.Compare)
		// 带参数路径最后注册
		selectRouter.GET("/:id", selectApi.GetDetail)
		selectRouter.POST("/:id/test", selectApi.Test)
	}
}
