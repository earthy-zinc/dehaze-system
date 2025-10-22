package router

import (
	"github.com/earthyzinc/dehaze-go/api"
	"github.com/earthyzinc/dehaze-go/middleware"
	"github.com/gin-gonic/gin"
)

type AlgorithmRouter struct{}

func (algorithmRouter *AlgorithmRouter) InitAlgorithmRouter(routerGroup *gin.RouterGroup) {
	algorithmApi := api.ApiGroupApp.AlgorithmApi
	algorithmRouterGroup := routerGroup.Group("/algorithm").
		Use(middleware.JWTAuth())

	{
		algorithmRouterGroup.GET("", algorithmApi.GetList)            // 获取算法树形表格
		algorithmRouterGroup.GET("/options", algorithmApi.GetOptions) // 获取模型下拉选项列表
		algorithmRouterGroup.GET("/:id", algorithmApi.GetById)        // 根据ID获取算法信息
		algorithmRouterGroup.POST("", algorithmApi.Add)               // 新增算法
		algorithmRouterGroup.PUT("/:id", algorithmApi.Update)         // 修改算法
		algorithmRouterGroup.DELETE("/:ids", algorithmApi.Delete)     // 删除算法
	}
}
