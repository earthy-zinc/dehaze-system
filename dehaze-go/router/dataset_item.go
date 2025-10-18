package router

import (
	"github.com/earthyzinc/dehaze-go/api"
	"github.com/earthyzinc/dehaze-go/middleware"
	"github.com/gin-gonic/gin"
)

type DatasetItemRouter struct{}

func (datasetItemRouter *DatasetItemRouter) InitDatasetItemRouter(routerGroup *gin.RouterGroup) {
	datasetItemApi := api.ApiGroupApp.SysDatasetItemApi
	datasetItemRouterGroup := routerGroup.Group("/dataset/item").
		Use(middleware.JWTAuth())

	{
		datasetItemRouterGroup.POST("", datasetItemApi.CreateDatasetItem)    // 新增数据项
		datasetItemRouterGroup.PUT("", datasetItemApi.UpdateDatasetItem)     // 修改数据项
		datasetItemRouterGroup.DELETE("", datasetItemApi.DeleteDatasetItem)  // 删除数据项
	}
}