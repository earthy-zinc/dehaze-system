package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterDatasetRoutes(rg *gin.RouterGroup, datasetApi *api.SysDatasetApi) {
	datasetRouterGroup := rg.Group("/datasets")

	{
		// 读操作 - 无需额外权限（静态路由须在 /:id 之前声明，避免被路径参数吞掉）
		datasetRouterGroup.GET("", datasetApi.GetDatasetList)
		datasetRouterGroup.GET("/tree", datasetApi.GetDatasetTree)
		datasetRouterGroup.GET("/options", datasetApi.GetDatasetOptions)
		datasetRouterGroup.GET("/evaluation-options", datasetApi.GetEvaluationOptions)
		datasetRouterGroup.GET("/children/:parentId", datasetApi.GetDatasetChildren)
		datasetRouterGroup.GET("/:id", datasetApi.GetDatasetById)

		// 写操作 - 需要权限校验（python 侧无防重复提交中间件，同级重名由唯一性校验兜底）
		datasetRouterGroup.POST("", middleware.Permission("sys:dataset:add"), datasetApi.SaveDataset)
		datasetRouterGroup.PUT("/:id", middleware.Permission("sys:dataset:edit"), datasetApi.UpdateDataset)
		datasetRouterGroup.DELETE("/:id", middleware.Permission("sys:dataset:delete"), datasetApi.DeleteDataset)
		datasetRouterGroup.DELETE("/batch", middleware.Permission("sys:dataset:delete"), datasetApi.BatchDeleteDatasets)
	}
}
