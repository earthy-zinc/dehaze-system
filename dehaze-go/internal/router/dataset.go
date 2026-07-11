package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

func RegisterDatasetRoutes(rg *gin.RouterGroup, datasetApi *api.SysDatasetApi) {
	datasetRouterGroup := rg.Group("/datasets")

	{
		datasetRouterGroup.GET("", datasetApi.GetDatasetList)
		datasetRouterGroup.GET("/options", datasetApi.GetDatasetOptions)
		datasetRouterGroup.DELETE("/batch", datasetApi.BatchDeleteDatasets)
		datasetRouterGroup.GET("/children/:parentId", datasetApi.GetDatasetChildren)
		datasetRouterGroup.POST("", datasetApi.SaveDataset)
		datasetRouterGroup.GET("/:id", datasetApi.GetDatasetById)
		datasetRouterGroup.PUT("/:id", datasetApi.UpdateDataset)
		datasetRouterGroup.DELETE("/:id", datasetApi.DeleteDataset)
	}
}
