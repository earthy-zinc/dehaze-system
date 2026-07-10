package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

func RegisterAlgorithmRoutes(rg *gin.RouterGroup, algorithmApi *api.AlgorithmApi) {
	algorithmRouterGroup := rg.Group("/algorithms")

	{
		algorithmRouterGroup.GET("", algorithmApi.GetList)            // 获取算法树形表格
		algorithmRouterGroup.GET("/options", algorithmApi.GetOptions) // 获取模型下拉选项列表
		algorithmRouterGroup.GET("/:id", algorithmApi.GetById)        // 根据ID获取算法信息
		algorithmRouterGroup.POST("", algorithmApi.Add)               // 新增算法
		algorithmRouterGroup.PUT("/:id", algorithmApi.Update)         // 修改算法
		algorithmRouterGroup.DELETE("", algorithmApi.Delete)           // 删除算法
	}
}
