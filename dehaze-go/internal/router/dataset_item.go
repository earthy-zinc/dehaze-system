package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterDatasetItemRoutes(rg *gin.RouterGroup, datasetItemApi *api.SysDatasetItemApi) {
	datasetItemRouterGroup := rg.Group("/dataset-items")

	{
		// 读操作 - 无需额外权限
		datasetItemRouterGroup.GET("", datasetItemApi.GetDatasetItems)        // 分页查询数据项列表
		datasetItemRouterGroup.GET("/:id", datasetItemApi.GetDatasetItemById) // 获取数据项详情

		// 写操作 - 需要权限校验 + 防重复提交（数据集编辑权限）
		datasetItemRouterGroup.POST("", middleware.Permission("sys:dataset:edit"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 3}), datasetItemApi.CreateDatasetItem)                       // 创建空数据项
		datasetItemRouterGroup.POST("/upload", middleware.Permission("sys:dataset:edit"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 5}), datasetItemApi.CreateDatasetItemWithImages)      // 创建数据项并上传配对图片
		datasetItemRouterGroup.POST("/batch", middleware.Permission("sys:dataset:edit"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 5}), datasetItemApi.BatchCreateDatasetItemsWithImages) // 批量创建数据项并上传图片
		datasetItemRouterGroup.PUT("/:id", middleware.Permission("sys:dataset:edit"), datasetItemApi.UpdateDatasetItem)                                                                                   // 修改数据项
		datasetItemRouterGroup.DELETE("/:id", middleware.Permission("sys:dataset:delete"), datasetItemApi.DeleteDatasetItem)                                                                              // 删除数据项
		datasetItemRouterGroup.DELETE("/batch", middleware.Permission("sys:dataset:delete"), datasetItemApi.BatchDeleteDatasetItems)                                                                      // 批量删除数据项
	}
}
