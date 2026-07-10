package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterSysMenuRoutes(rg *gin.RouterGroup, sysMenuApi *api.SysMenuApi) gin.IRoutes {
	sysMenuRouter := rg.Group("menus")
	{
		// 读操作 - 无权限校验
		sysMenuRouter.GET("", sysMenuApi.ListMenus)
		sysMenuRouter.GET("options", sysMenuApi.ListMenuOptions)
		sysMenuRouter.GET("routes", sysMenuApi.ListRoutes)
		sysMenuRouter.GET(":id/form", sysMenuApi.GetMenuForm)

		// 写操作 - 需要权限校验
		sysMenuRouter.POST("", middleware.Permission("sys:menu:add"), sysMenuApi.SaveMenu)
		sysMenuRouter.PUT(":id", middleware.Permission("sys:menu:edit"), sysMenuApi.UpdateMenu)
		sysMenuRouter.DELETE(":id", middleware.Permission("sys:menu:delete"), sysMenuApi.DeleteMenu)
		sysMenuRouter.PATCH(":id", middleware.Permission("sys:menu:edit"), sysMenuApi.UpdateMenuVisible)
	}
	return sysMenuRouter
}
