package router

import (
	"github.com/earthyzinc/clarity-go/internal/api"
	"github.com/earthyzinc/clarity-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterDatasetItemRoutes(rg *gin.RouterGroup, datasetItemApi *api.SysDatasetItemApi) {
	datasetItemRouterGroup := rg.Group("/dataset-items")

	{
		// 读操作 - 无需额外权限
		datasetItemRouterGroup.GET("", datasetItemApi.GetDatasetItems)        // 分页查询数据项列表
		datasetItemRouterGroup.GET("/:id", datasetItemApi.GetDatasetItemById) // 获取数据项详情

		// 写操作 - 需要权限校验（数据集编辑权限）。
		// 不挂防重：本组为"同一数据集下连续添加多项"的合法连续操作场景，
		// 且上传请求体为 multipart（无法有效摘要），纳入防重只会误伤。
		datasetItemRouterGroup.POST("", middleware.Permission("sys:dataset:edit"), datasetItemApi.CreateDatasetItem)                       // 创建空数据项
		datasetItemRouterGroup.POST("/upload", middleware.Permission("sys:dataset:edit"), datasetItemApi.CreateDatasetItemWithImages)      // 创建数据项并上传配对图片
		datasetItemRouterGroup.POST("/batch", middleware.Permission("sys:dataset:edit"), datasetItemApi.BatchCreateDatasetItemsWithImages) // 批量创建数据项并上传图片
		datasetItemRouterGroup.PUT("/:id", middleware.Permission("sys:dataset:edit"), datasetItemApi.UpdateDatasetItem)                    // 修改数据项
		datasetItemRouterGroup.DELETE("/:id", middleware.Permission("sys:dataset:delete"), datasetItemApi.DeleteDatasetItem)               // 删除数据项
		datasetItemRouterGroup.DELETE("/batch", middleware.Permission("sys:dataset:delete"), datasetItemApi.BatchDeleteDatasetItems)       // 批量删除数据项
	}
}
