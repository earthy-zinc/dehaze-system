package router

import (
	"github.com/earthyzinc/dehaze-go/api"
	"github.com/earthyzinc/dehaze-go/middleware"
	"github.com/gin-gonic/gin"
)

type ItemFileRouter struct{}

func (itemFileRouter *ItemFileRouter) InitItemFileRouter(routerGroup *gin.RouterGroup) {
	itemFileApi := api.ApiGroupApp.SysItemFileApi
	itemFileRouterGroup := routerGroup.Group("/dataset/image").
		Use(middleware.JWTAuth())

	{
		itemFileRouterGroup.POST("", itemFileApi.AddImageById)      // 上传数据项图片
		itemFileRouterGroup.PUT("", itemFileApi.UpdateImageById)    // 修改数据项图片信息
		itemFileRouterGroup.DELETE("", itemFileApi.RemoveImageById) // 删除数据项图片
	}
}
