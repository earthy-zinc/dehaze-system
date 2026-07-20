package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterSysDictRoutes(rg *gin.RouterGroup, sysDictApi *api.SysDictApi) gin.IRoutes {
	sysDictRouter := rg.Group("dict")
	{
		// 字典类型相关路由（优先级高，避免参数冲突）
		sysDictRouter.GET("types/page", sysDictApi.GetDictTypePage)
		sysDictRouter.GET("types/:id/form", sysDictApi.GetDictTypeForm)
		sysDictRouter.POST("types", middleware.Permission("sys:dict:type:add"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 3}), sysDictApi.SaveDictType)
		sysDictRouter.PUT("types/:id", middleware.Permission("sys:dict:type:edit"), sysDictApi.UpdateDictType)
		sysDictRouter.DELETE("types/:ids", middleware.Permission("sys:dict:type:delete"), sysDictApi.DeleteDictTypes)

		// 字典数据项相关路由
		sysDictRouter.GET("page", sysDictApi.GetDictPage)
		sysDictRouter.POST("", middleware.Permission("sys:dict:data:add"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 3}), sysDictApi.SaveDict)
		sysDictRouter.GET(":id/options", sysDictApi.ListDictOptions)
		sysDictRouter.GET(":id/form", sysDictApi.GetDictForm)
		sysDictRouter.PUT(":id", middleware.Permission("sys:dict:data:edit"), sysDictApi.UpdateDict)
		sysDictRouter.DELETE(":ids", middleware.Permission("sys:dict:data:delete"), sysDictApi.DeleteDict)
	}
	return sysDictRouter
}
