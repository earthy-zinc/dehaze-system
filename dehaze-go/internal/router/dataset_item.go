package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

type DatasetItemRouter struct{}

func (datasetItemRouter *DatasetItemRouter) InitDatasetItemRouter(routerGroup *gin.RouterGroup) {
	datasetItemApi := api.ApiGroupApp.SysDatasetItemApi
	datasetItemRouterGroup := routerGroup.Group("/dataset-items").
		Use(middleware.JWTAuth())

	{
		datasetItemRouterGroup.GET("/:id", datasetItemApi.GetDatasetItemById) // 获取数据项详情
		datasetItemRouterGroup.GET("", datasetItemApi.GetDatasetItems)        // 分页查询数据项列表
		datasetItemRouterGroup.POST("", datasetItemApi.CreateDatasetItem)     // 新增数据项
		datasetItemRouterGroup.PUT("", datasetItemApi.UpdateDatasetItem)      // 修改数据项
		datasetItemRouterGroup.DELETE("", datasetItemApi.DeleteDatasetItem)   // 删除数据项
	}
}
