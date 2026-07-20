package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterDatasetRoutes(rg *gin.RouterGroup, datasetApi *api.SysDatasetApi) {
	datasetRouterGroup := rg.Group("/datasets")

	{
		// 读操作 - 无需额外权限
		datasetRouterGroup.GET("", datasetApi.GetDatasetList)
		datasetRouterGroup.GET("/tree", datasetApi.GetDatasetTree)
		datasetRouterGroup.GET("/options", datasetApi.GetDatasetOptions)
		datasetRouterGroup.GET("/children/:parentId", datasetApi.GetDatasetChildren)
		datasetRouterGroup.GET("/:id", datasetApi.GetDatasetById)

		// 写操作 - 需要权限校验 + 防重复提交
		datasetRouterGroup.POST("", middleware.Permission("sys:dataset:add"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 3}), datasetApi.SaveDataset)
		datasetRouterGroup.PUT("/:id", middleware.Permission("sys:dataset:edit"), datasetApi.UpdateDataset)
		datasetRouterGroup.DELETE("/:id", middleware.Permission("sys:dataset:delete"), datasetApi.DeleteDataset)
		datasetRouterGroup.DELETE("/batch", middleware.Permission("sys:dataset:delete"), datasetApi.BatchDeleteDatasets)
	}
}
