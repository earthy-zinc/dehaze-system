package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

func RegisterDatasetRoutes(rg *gin.RouterGroup, datasetApi *api.SysDatasetApi) {
	datasetRouterGroup := rg.Group("/datasets")

	{
		datasetRouterGroup.GET("", datasetApi.GetDatasetList)                 // 获取数据集列表（支持树形）
		datasetRouterGroup.GET("/options", datasetApi.GetDatasetOptions)      // 获取数据集下拉选项
		datasetRouterGroup.GET("/:id", datasetApi.GetDatasetById)             // 获取数据集详情
		datasetRouterGroup.GET("/:id/stats", datasetApi.GetDatasetStatistics) // 获取数据集统计信息
		datasetRouterGroup.POST("", datasetApi.SaveDataset)                   // 新增数据集
		datasetRouterGroup.PUT("/:id", datasetApi.UpdateDataset)              // 修改数据集
		datasetRouterGroup.DELETE("/:id", datasetApi.DeleteDataset)           // 删除单个数据集
		datasetRouterGroup.DELETE("/batch", datasetApi.BatchDeleteDatasets)   // 批量删除数据集
	}
}
