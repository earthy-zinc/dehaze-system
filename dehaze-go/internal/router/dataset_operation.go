// ... existing code ...
package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

type DatasetOperationRouter struct{}

func (datasetOperationRouter *DatasetOperationRouter) InitDatasetOperationRouter(routerGroup *gin.RouterGroup) {
	datasetOperationApi := api.ApiGroupApp.DatasetOperationApi
	datasetOperationRouterGroup := routerGroup.Group("/dataset/operations").
		Use(middleware.JWTAuth())

	{
		// 创建数据项（带图片）
		datasetOperationRouterGroup.POST("/items", datasetOperationApi.CreateDatasetItemWithImages)
		// 批量创建数据项（带图片）
		datasetOperationRouterGroup.POST("/items/batch", datasetOperationApi.BatchCreateDatasetItemsWithImages)
		// 级联删除数据项
		datasetOperationRouterGroup.DELETE("/items/:itemId", datasetOperationApi.DeleteDatasetItemCascade)
		// 批量删除数据集
		datasetOperationRouterGroup.POST("/batch", datasetOperationApi.BatchDeleteDatasets)
	}
}
