package router

import (
	"github.com/earthyzinc/dehaze-go/api"
	"github.com/earthyzinc/dehaze-go/middleware"
	"github.com/gin-gonic/gin"
)

type DatasetRouter struct{}

func (datasetRouter *DatasetRouter) InitDatasetRouter(routerGroup *gin.RouterGroup) {
	datasetApi := api.ApiGroupApp.SysDatasetApi
	datasetRouterGroup := routerGroup.Group("/dataset").
		Use(middleware.JWTAuth())

	{
		datasetRouterGroup.GET("", datasetApi.GetDatasetList)            // 获取数据集树形列表
		datasetRouterGroup.GET("/options", datasetApi.GetDatasetOptions) // 获取数据集下拉选项
		datasetRouterGroup.GET("/:id/form", datasetApi.GetDatasetForm)   // 获取数据集表单数据
		datasetRouterGroup.POST("", datasetApi.SaveDataset)              // 新增数据集
		datasetRouterGroup.PUT("/:id", datasetApi.UpdateDataset)         // 修改数据集
		datasetRouterGroup.DELETE("", datasetApi.DeleteDatasets)         // 删除数据集
	}
}
