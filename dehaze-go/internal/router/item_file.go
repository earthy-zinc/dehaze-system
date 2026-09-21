package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterItemFileRoutes(rg *gin.RouterGroup, itemFileApi *api.SysItemFileApi) {
	itemFileRouterGroup := rg.Group("/item-files")

	{
		// 读操作 - 无需额外权限
		itemFileRouterGroup.GET("/:id", itemFileApi.GetItemFileById) // 获取图片详细信息

		// 写操作 - 权限码对齐数据项管理与权限种子（sys:file:edit/sys:file:delete 未在 sys_menu 中定义）
		itemFileRouterGroup.POST("", middleware.Permission("sys:dataset:edit"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 5}), itemFileApi.AddImageById) // 上传数据项图片
		itemFileRouterGroup.PUT("/:id", middleware.Permission("sys:dataset:edit"), itemFileApi.UpdateImageById)                                                          // 修改图片信息
		itemFileRouterGroup.DELETE("/:id", middleware.Permission("sys:dataset:delete"), itemFileApi.RemoveImageById)                                                     // 删除图片
		itemFileRouterGroup.DELETE("/batch", middleware.Permission("sys:dataset:delete"), itemFileApi.BatchRemoveImages)                                                 // 批量删除图片
	}
}
