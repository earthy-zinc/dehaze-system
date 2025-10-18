package router

import (
	"github.com/earthyzinc/dehaze-go/api"
	"github.com/gin-gonic/gin"
)

type SysDictRouter struct{}

func (r *SysDictRouter) InitSysDictRouter(Router *gin.RouterGroup) (R gin.IRoutes) {
	sysDictRouter := Router.Group("dict")
	sysDictApi := api.ApiGroupApp.SysDictApi
	{
		// 字典数据项相关路由
		sysDictRouter.GET("page", sysDictApi.GetDictPage)
		sysDictRouter.GET(":id/form", sysDictApi.GetDictForm)
		sysDictRouter.POST("", sysDictApi.SaveDict)
		sysDictRouter.PUT(":id", sysDictApi.UpdateDict)
		sysDictRouter.DELETE(":ids", sysDictApi.DeleteDict)
		sysDictRouter.GET(":typeCode/options", sysDictApi.ListDictOptions)

		// 字典类型相关路由
		sysDictRouter.GET("types/page", sysDictApi.GetDictTypePage)
		sysDictRouter.GET("types/:id/form", sysDictApi.GetDictTypeForm)
		sysDictRouter.POST("types", sysDictApi.SaveDictType)
		sysDictRouter.PUT("types/:id", sysDictApi.UpdateDictType)
		sysDictRouter.DELETE("types/:ids", sysDictApi.DeleteDictTypes)
	}
	return sysDictRouter
}