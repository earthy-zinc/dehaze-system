package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

type SysDictRouter struct{}

func (r *SysDictRouter) InitSysDictRouter(Router *gin.RouterGroup) (R gin.IRoutes) {
	sysDictRouter := Router.Group("dict")
	sysDictApi := api.ApiGroupApp.SysDictApi
	{
		// 字典类型相关路由（优先级高，避免参数冲突）
		sysDictRouter.GET("types/page", sysDictApi.GetDictTypePage)
		sysDictRouter.GET("types/:id/form", sysDictApi.GetDictTypeForm)
		sysDictRouter.POST("types", sysDictApi.SaveDictType)
		sysDictRouter.PUT("types/:id", sysDictApi.UpdateDictType)
		sysDictRouter.DELETE("types/:ids", sysDictApi.DeleteDictTypes)

		// 字典数据项相关路由
		sysDictRouter.GET("page", sysDictApi.GetDictPage)
		sysDictRouter.POST("", sysDictApi.SaveDict)
		sysDictRouter.GET("options/:typeCode", sysDictApi.ListDictOptions)
		sysDictRouter.GET(":id/form", sysDictApi.GetDictForm)
		sysDictRouter.PUT(":id", sysDictApi.UpdateDict)
		sysDictRouter.DELETE(":ids", sysDictApi.DeleteDict)
	}
	return sysDictRouter
}
