package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

func RegisterDatasetItemRoutes(rg *gin.RouterGroup, datasetItemApi *api.SysDatasetItemApi) {
	datasetItemRouterGroup := rg.Group("/dataset-items")

	{
		datasetItemRouterGroup.GET("", datasetItemApi.GetDatasetItems)                          // 分页查询数据项列表
		datasetItemRouterGroup.GET("/:id", datasetItemApi.GetDatasetItemById)                   // 获取数据项详情
		datasetItemRouterGroup.POST("", datasetItemApi.CreateDatasetItem)                       // 创建空数据项
		datasetItemRouterGroup.POST("/upload", datasetItemApi.CreateDatasetItemWithImages)      // 创建数据项并上传配对图片
		datasetItemRouterGroup.POST("/batch", datasetItemApi.BatchCreateDatasetItemsWithImages) // 批量创建数据项并上传图片
		datasetItemRouterGroup.PUT("/:id", datasetItemApi.UpdateDatasetItem)                    // 修改数据项
		datasetItemRouterGroup.DELETE("/:id", datasetItemApi.DeleteDatasetItem)                 // 删除数据项
		datasetItemRouterGroup.DELETE("/batch", datasetItemApi.BatchDeleteDatasetItems)         // 批量删除数据项
	}
}
