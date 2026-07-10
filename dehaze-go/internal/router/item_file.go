package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

func RegisterItemFileRoutes(rg *gin.RouterGroup, itemFileApi *api.SysItemFileApi) {
	itemFileRouterGroup := rg.Group("/item-files")

	{
		itemFileRouterGroup.GET("/:id", itemFileApi.GetItemFileById)        // 获取图片详细信息
		itemFileRouterGroup.POST("", itemFileApi.AddImageById)              // 上传数据项图片
		itemFileRouterGroup.PUT("/:id", itemFileApi.UpdateImageById)        // 修改图片信息
		itemFileRouterGroup.DELETE("/:id", itemFileApi.RemoveImageById)     // 删除图片
		itemFileRouterGroup.DELETE("/batch", itemFileApi.BatchRemoveImages) // 批量删除图片
	}
}
